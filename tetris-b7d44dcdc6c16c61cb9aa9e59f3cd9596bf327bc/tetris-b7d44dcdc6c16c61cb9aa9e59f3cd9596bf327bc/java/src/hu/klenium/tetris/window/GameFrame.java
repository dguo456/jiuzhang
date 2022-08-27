package hu.klenium.tetris.window;

import hu.klenium.tetris.TetrisGame;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;

import static hu.klenium.tetris.UserCommand.*;

public class GameFrame {
    private Canvas boardCanvas;
    private Canvas tetrominoCanvas;
    private Scene scene;
    public Canvas getBoardCanvas() {
        return boardCanvas;
    }
    public Canvas getTetrominoCanvas() {
        return tetrominoCanvas;
    }
    public GameFrame(Scene scene) {
        this.scene = scene;
        StackPane pane = new StackPane();
        boardCanvas = new Canvas();
        tetrominoCanvas = new Canvas();
        pane.getChildren().addAll(boardCanvas, tetrominoCanvas);
        ((HBox) scene.getRoot()).getChildren().add(pane);
    }
    public void registerEventListeners(TetrisGame game) {
        scene.addEventHandler(KeyEvent.KEY_PRESSED, event -> {
            switch (event.getCode()) {
                case W: game.handleCommand(ROTATE); break;
                case A: game.handleCommand(MOVE_LEFT); break;
                case S: game.handleCommand(MOVE_DOWN); break;
                case D: game.handleCommand(MOVE_RIGHT); break;
                case SPACE: game.handleCommand(DROP); break;
            }
        });
    }
    public void setSize(int width, int height) {
        boardCanvas.setWidth(width);
        boardCanvas.setHeight(height);
        tetrominoCanvas.setWidth(width);
        tetrominoCanvas.setHeight(height);
        scene.getWindow().sizeToScene();
    }
}