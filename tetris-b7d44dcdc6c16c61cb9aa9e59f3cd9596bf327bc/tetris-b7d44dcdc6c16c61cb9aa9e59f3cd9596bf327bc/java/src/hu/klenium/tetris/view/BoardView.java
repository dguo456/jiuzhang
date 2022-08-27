package hu.klenium.tetris.view;

import hu.klenium.tetris.window.GameFrame;

public class BoardView extends CanvasView {
    public BoardView(GameFrame frame, int squareSize) {
        super(frame.getBoardCanvas(), squareSize);
    }
    public void update(SquareView[][] data) {
        clear();
        int height = data.length;
        int width = data[0].length;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (data[i][j] != null)
                    data[i][j].update(context, j, i, squareSize);
            }
        }
    }
}