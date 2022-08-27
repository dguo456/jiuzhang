package hu.klenium.tetris.window;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

public class MainApplication extends Application {
    private static Runnable readyTask;
    private static Scene scene;
    public static void init(Runnable task) {
        readyTask = task;
        launch();
    }
    public static GameFrame createFrame() {
        return new GameFrame(scene);
    }
    @Override public void start(Stage primaryStage) {
        HBox root = new HBox();
        root.setStyle("-fx-background-color: rgb(23, 23, 23);");
        scene = new Scene(root);
        primaryStage.setTitle("Tetris");
        primaryStage.setResizable(false);
        primaryStage.setScene(scene);
        primaryStage.show();
        root.requestFocus();
        readyTask.run();
    }
}