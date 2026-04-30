import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, Normal, OneHotCategoricalStraightThrough
import numpy as np
import os

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderConv, DecoderConv, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
import imageio


class Dreamer:
    def __init__(self, observationShape, actionSize, actionLow, actionHigh, device, config):
        self.observationShape = observationShape
        self.actionSize = actionSize
        self.device = device
        self.rootConfig = config
        self.config = config.dreamer

        self.recurrentSize = self.config.recurrentSize
        self.latentSize = self.config.latentLength * self.config.latentClasses
        self.fullStateSize = self.recurrentSize + self.latentSize


        self.actor = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device, self.config.actor).to(self.device)
        self.critic = Critic(self.fullStateSize, self.config.critic).to(self.device)
        self.encoder = EncoderConv(observationShape, self.config.encodedObsSize, self.config.encoder).to(self.device)
        self.decoder = DecoderConv(self.fullStateSize, observationShape, self.config.decoder).to(self.device)
        self.recurrentModel = RecurrentModel(self.config.recurrentSize, self.latentSize, actionSize, self.config.recurrentModel).to(self.device)
        self.priorNet = PriorNet(self.config.recurrentSize, self.config.latentLength, self.config.latentClasses, self.config.priorNet).to(self.device)
        self.posteriorNet = PosteriorNet(self.config.recurrentSize + self.config.encodedObsSize, self.config.latentLength, self.config.latentClasses, self.config.posteriorNet).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize, self.config.reward).to(self.device)

        if self.config.useContinuationPrediction:
            self.continuePredictor = ContinueModel(self.fullStateSize, self.config.continuation).to(self.device)

        self.buffer = ReplayBuffer(observationShape, actionSize, self.config.buffer, device)
        self.valueMoments = Moments(device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                        list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer = torch.optim.Adam(self.worldModelParameters, lr=self.config.worldModelLR)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actorLR)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.criticLR)

        self.totalEpisodes = 0
        self.totalEnvSteps = 0
        self.totalGradientSteps = 0


    def worldModelTraining(self, data):
        data.encodedObservation = self.encoder(data.observation.view(-1, *self.observationShape)).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState, previousLatentState = torch.zeros(len(data.action), self.recurrentSize, device=self.device), torch.zeros(len(data.action), self.latentSize, device=self.device)

        recurrenStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState = self.recurrentModel(previousRecurrentState, previousLatentState, data.action[:, t-1])
            _, priorLogits = self.priorNet(recurrentState)
            posterior, posteriorLogits = self.posteriorNet(torch.cat((recurrentState, data.encodedObservation[:, t]), dim=-1))


            recurrenStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState, previousLatentState = recurrentState, posterior

        recurrenStates = torch.stack(recurrenStates, dim=1)
        priorsLogits = torch.stack(priorsLogits, dim=1)
        posteriors = torch.stack(posteriors, dim=1)
        posteriorsLogits = torch.stack(posteriorsLogits, dim=1)
        fullStates = torch.cat((recurrenStates, posteriors), dim=-1)

        reconstructionMeans = self.decoder(fullStates.view(-1, self.fullStateSize)).view(self.config.batchSize, self.config.batchLength-1, *self.observationShape)
        reconstructionDistibution = Independent(Normal(reconstructionMeans, 1), len(self.observationShape))
        reconstructionLoss = -reconstructionDistibution.log_prob(data.observation[:, 1:]).mean()

        rewardDistirbution = self.rewardPredictor(fullStates)
        rewardLoss = -rewardDistirbution.log_prob(data.reward[:, 1:].squeeze(-1)).mean()

        priorDistribution = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits), 1)
        priorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()), 1)
        posteriorDistribution = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach()), 1)
        
        priorLoss = kl_divergence(posteriorDistributionSG, priorDistribution)
        posteriorLoss = kl_divergence(posteriorDistribution, priorDistributionSG)
        freeNats = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss = self.config.betaPrior * torch.max(priorLoss, freeNats)
        posteriorLoss = self.config.betaPosterior * torch.max(posteriorLoss, freeNats)
        klLoss = (priorLoss + posteriorLoss).mean()

        worldModelLoss = reconstructionLoss + rewardLoss + klLoss


        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss = nn.BCELoss()(continueDistribution.probs, 1 - data.done[:, 1:].squeeze(-1))
            worldModelLoss += continueLoss.mean()
        
        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior) * self.config.freeNats

        metrics = {
            'worldModelLoss': worldModelLoss.item(),
            'reconstructionLoss': reconstructionLoss.item(),
            'rewardPredictorLoss': rewardLoss.item(),
            'klLoss': klLoss.item() - klLossShiftForGraphing}
        return fullStates.view(-1, self.fullStateSize).detach(), metrics
    

    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize),-1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)
        fullStates = torch.stack(fullStates, dim=1)
        logprobs = torch.stack(logprobs[1:], dim=1)
        entropies = torch.stack(entropies[1:], dim=1)

        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values = self.critic(fullStates).mean
        continues = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)
        lambdaValues = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages = (lambdaValues - values[:, :-1]) / inverseScale

        actorLoss = -torch.mean(advantages.detach() * logprobs + self.config.entropyScale * entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions = self.critic(fullStates[:, :-1].detach())
        criticLoss = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step() 

        metrics = {
            'actorLoss': actorLoss.item(),
            'criticLoss': criticLoss.item(),
            'entropies' : entropies.mean().item(),
            'logprobs': logprobs.mean().item(),
            'advantages': advantages.mean().item(),
            'criticValues': values.mean().item()}
        return metrics
    
    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, seed=None, evaluation=False, saveVideo=False, filename="videos/unnmaedVideo", fps=30, macroBlockSize=16):
        scores = []
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed=(seed + self.totalEpisodes if seed else None ))
            encodedObservation = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))

            currentScore, stepcount, done, frames = 0, 0, False, []

            while not done:
                recurrentState = self.recurrentModel(recurrentState, latentState, action)
                latentState, _ = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), dim=-1))

                action = self.actor(torch.cat((recurrentState, latentState), -1))

                actionNumpy = action.cpu().numpy().reshape(-1)

                nextObservation, reward, done = env.step(actionNumpy)

                if not evaluation:
                    self.buffer.add(observation, actionNumpy, reward, nextObservation, done)

                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight, targetWidth = (macroBlockSize - 1 + (frame.shape[0]) )// macroBlockSize * macroBlockSize, (macroBlockSize -1 + frame.shape[1]) // macroBlockSize * macroBlockSize
                    frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
                observation = nextObservation

                currentScore += reward
                stepcount += 1
                if done:
                    scores.append(currentScore)

                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepcount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.04f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    
                    break
            return sum(scores)/numEpisodes if numEpisodes else None

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'recurrentModel': self.recurrentModel.state_dict(),
            'priorNet': self.priorNet.state_dict(),
            'posteriorNet': self.posteriorNet.state_dict(),
            'rewardPredictor': self.rewardPredictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'worldModelOptimizer': self.worldModelOptimizer.state_dict(),
            'actorOptimizer': self.actorOptimizer.state_dict(),
            'criticOptimizer': self.criticOptimizer.state_dict(),
            'totalEpisodes': self.totalEpisodes,
            'totalEnvSteps': self.totalEnvSteps,
            'totalGradientSteps': self.totalGradientSteps
        }   

        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpointPath}")

        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']

        if self.config.useContinuationPrediction and 'continuePredictor' in checkpoint:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])                     