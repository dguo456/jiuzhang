package hu.klenium.tetris.view;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class SquareView {
    private Color color;
    public SquareView(int type) {
        this.color = colors[type];
    }
    public void update(GraphicsContext context, int x, int y, int size) {
        context.setFill(color);
        context.fillRect(x * size, y * size, size, size);
    }
    private static Color[] colors = new Color[] {
            Color.rgb(0, 200, 100),
            Color.rgb(70, 130, 0),
            Color.rgb(160, 220, 0),
            Color.rgb(250, 230, 60),
            Color.rgb(250, 180, 50),
            Color.rgb(230, 100, 0),
            Color.rgb(200, 0, 0)
    };
}