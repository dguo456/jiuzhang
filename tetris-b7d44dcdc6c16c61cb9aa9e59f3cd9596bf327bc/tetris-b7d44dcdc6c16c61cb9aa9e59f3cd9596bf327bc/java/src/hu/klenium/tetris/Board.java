package hu.klenium.tetris;

import hu.klenium.tetris.view.BoardView;
import hu.klenium.tetris.view.SquareView;

public class Board {
    private final int rows;
    private final int columns;
    private SquareView[][] board;
    private final BoardView view;

    public Board(int rows, int cols, BoardView view) {
        this.rows = rows;
        this.columns = cols;
        this.board = new SquareView[rows][cols];
        this.view = view;
        updateView();
    }
    public int getHeight() {
        return rows;
    }
    public int getWidth() {
        return columns;
    }

    public boolean canAddTetromino(Tetromino tetromino, int fromX, int fromY) {
        SquareView[][] data = tetromino.getPolyominoData();
        int height = data.length;
        int width = data[0].length;
        if (fromX < 0 || fromX + width > columns ||
            fromY < 0 || fromY + height > rows)
            return false;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (data[i][j] != null && board[fromY + i][fromX + j] != null)
                    return false;
            }
        }
        return true;
    }
    public void addTetromino(Tetromino tetromino) {
        SquareView[][] data = tetromino.getPolyominoData();
        int x = tetromino.getPosX();
        int y = tetromino.getPosY();
        int height = data.length;
        int width = data[0].length;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (data[i][j] != null)
                    board[y + i][x + j] = data[i][j];
            }
        }
        updateView();
    }
    public void removeFullRows() {
        boolean isRowFull;
        for (int i = 0; i < rows; ++i) {
            isRowFull = true;
            for (int j = 0; j < columns && isRowFull; ++j) {
                if (board[i][j] == null)
                    isRowFull = false;
            }
            if (isRowFull) {
                for (int j = i; j > 0; --j)
                    System.arraycopy(board[j - 1], 0, board[j], 0, columns);
                for (int j = 0; j < columns; ++j)
                    board[0][j] = null;
            }
        }
        updateView();
    }

    private void updateView() {
        view.update(board);
    }
}