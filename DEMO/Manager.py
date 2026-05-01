import Input
import Model

# manager = input.Manager()
# manager.inputHandler()
# manager.printCardsList()


def main() -> None:
    model = Model.Model()
    io_manager = Input.IOput_Manager()

    io_manager.inputHandler()
    inputData = io_manager.getDFCardsList()  # returns input as dataframe
    model.checkChance(inputData)  # interprets data then stores win chance

    io_manager.printWinChance(model.getConfidence(), model.getPrediction())


if __name__ == "__main__":
    main()
 




