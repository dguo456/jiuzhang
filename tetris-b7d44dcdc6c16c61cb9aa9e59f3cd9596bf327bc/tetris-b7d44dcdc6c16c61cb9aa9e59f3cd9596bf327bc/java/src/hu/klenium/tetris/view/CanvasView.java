package hu.klenium.tetris.view;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

public class CanvasView {
    protected GraphicsContext context;
    protected int squareSize;
    private int contextWidth;
    private int contextHeight;
    public CanvasView(Canvas canvas, int squareSize) {
        context = canvas.getGraphicsContext2D();
        contextWidth = (int) canvas.getWidth();
        contextHeight = (int) canvas.getHeight();
        this.squareSize = squareSize;
    }
    public void clear() {
        context.clearRect(0, 0, contextWidth, contextHeight);
    }
}