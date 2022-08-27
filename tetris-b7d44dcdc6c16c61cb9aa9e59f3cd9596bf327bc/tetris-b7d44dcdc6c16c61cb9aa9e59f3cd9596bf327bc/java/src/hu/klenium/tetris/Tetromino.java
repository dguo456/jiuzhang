package hu.klenium.tetris;

import hu.klenium.tetris.view.SquareView;
import hu.klenium.tetris.view.TetrominoView;

public class Tetromino {
    private Board board;
    private SquareView[][][] partsData;
    private TetrominoView view;
    private int currentX = 0;
    private int currentY = 0;
    private int rotation;
    private int width;
    private int height;

    private Tetromino(int type, TetrominoView view, Board board) {
        this.view = view;
        this.board = board;
        partsData = TetrominoDataSource.getData(type);
        setRotation(0);
    }
    public static Tetromino createAtCenter(int type, TetrominoView view, Board board) {
        Tetromino tetromino = new Tetromino(type, view, board);
        int x = (int) Math.ceil((board.getWidth() - tetromino.width) / 2);
        boolean moved = tetromino.tryMove(x, 0);
        if (!moved) {
            tetromino.dispose();
            return null;
        }
        return tetromino;
    }
    public void dispose() {
        view.clear();
    }

    public SquareView[][] getPolyominoData() {
        return partsData[rotation];
    }
    public int getPosX() {
        return currentX;
    }
    public int getPosY() {
        return currentY;
    }

    public boolean rotateRight() {
        int nextRotation = (rotation + 1) % 4;
        boolean canRotate = false;
        int oldRotation = rotation;
        setRotation(nextRotation);
        if (canMoveTo(0, 0))
            canRotate = true;
        else {
            for (int i = 1; i < width && !canRotate; ++i) {
                if (canMoveTo(-i, 0)) {
                    currentX -= i;
                    canRotate = true;
                }
            }
        }
        if (!canRotate)
            setRotation(oldRotation);
        else {
            setRotation(nextRotation);
            updateView();
        }
        return canRotate;
    }
    public boolean moveRight() {
        return tryMove(1, 0);
    }
    public boolean moveLeft() {
        return tryMove(-1, 0);
    }
    public boolean moveDown() {
        return tryMove(0, 1);
    }
    public void drop() {
        boolean movedDown;
        do {
            movedDown = moveDown();
        } while (movedDown);
    }

    private void setRotation(int rotation) {
        this.rotation = rotation % partsData.length;
        height = partsData[this.rotation].length;
        width = partsData[this.rotation][0].length;
    }
    private void updateView() {
        view.update(partsData[rotation], currentX, currentY);
    }
    private boolean tryMove(int x, int y) {
        boolean canSlide = canMoveTo(x, y);
        if (canSlide) {
            currentX += x;
            currentY += y;
            updateView();
        }
        return canSlide;
    }
    private boolean canMoveTo(int deltaX, int deltaY) {
        return board.canAddTetromino(this, currentX + deltaX, currentY + deltaY);
    }
}