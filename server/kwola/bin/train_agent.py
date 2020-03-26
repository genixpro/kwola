import kwola.tasks.TrainAgent
import mongoengine




def main():
    mongoengine.connect('kwola')
    kwola.tasks.TrainAgent.trainAgent()


