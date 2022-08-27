package hu.klenium.tetris;

import hu.klenium.tetris.window.MainApplication;

public class Main {
    public static void main(String[] args) {
        MainApplication.init(() -> {
            TetrisGame game = new TetrisGame();
            game.start();
        });
    }
}