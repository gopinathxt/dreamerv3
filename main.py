import gymnasium as gym
import torch
import argparse
import os
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils import saveLossesToCSV, ensureParentFolders
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(configFile):

    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}"
    checkpointToLoad = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename = os.path.join(config.folderNames.metricsFolder, runName)
    plotFilename = os.path.join(config.folderNames.plotsFolder, runName)
    checkpointFilenameBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    videoFilenameBase = os.path.join(config.folderNames.videosFolder, runName)

    env = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
    envEvaluation = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))


    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(f"Observation shape: {observationShape}, Action size: {actionSize}, Action low: {actionLow}, Action high: {actionHigh}")


    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)
    
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio

    metricsInterval = getattr(config, 'metricsLoggingInterval', getattr(config, 'checkpointInterval', None))
    saveCheckpoints = getattr(config, 'saveCheckpoints', getattr(config, 'saveCheckpoint', False))
    evaluationEpisodes = getattr(config, 'evaluationEpisodes', getattr(config, 'numEvaluationEpisodes', 1))

    for _ in range(1, 1 + iterationsNum):
        for _ in range(1, config.replayRatio + 1):
            data = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics = dreamer.worldModelTraining(data)
            actorCriticMetrics = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            if metricsInterval and dreamer.totalGradientSteps % metricsInterval == 0 and saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, evaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, videoFilename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

            mostRecentScore = dreamer.environmentInteraction(envEvaluation, config.numInteractionEpisodes, seed=config.seed)
            if config.saveMetrics:
                metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward": mostRecentScore}    
                saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | actorCriticMetrics)
                plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")      
    main(parser.parse_args().config)

