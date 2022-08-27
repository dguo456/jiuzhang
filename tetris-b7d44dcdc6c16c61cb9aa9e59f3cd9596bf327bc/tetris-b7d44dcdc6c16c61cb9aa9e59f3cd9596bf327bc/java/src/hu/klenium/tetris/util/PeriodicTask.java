package hu.klenium.tetris.util;

import java.util.Timer;
import java.util.TimerTask;

public class PeriodicTask {
    private Timer timer;
    private Runnable task;
    private int delay;
    public PeriodicTask(Runnable task, int delay) {
        this.task = task;
        this.delay = delay;
    }
    public void start() {
        if (timer == null) {
            timer = new Timer(true);
            timer.schedule(new TimerTask() {
                @Override public void run() {
                    task.run();
                }
            }, delay, delay);
        }
    }
    public void stop() {
        if (timer != null) {
            timer.cancel();
            timer = null;
        }
    }
    public void reset() {
        stop();
        start();
    }
}