package hu.klenium.tetris;

import hu.klenium.tetris.view.SquareView;

public class TetrominoDataSource {
    public static SquareView[][][] getData(int type) {
        String[][] masks = rawData[type];
        SquareView[][][] result = new SquareView[masks.length][][];
        for (int rotation = 0; rotation < masks.length; ++rotation) {
            int height = masks[rotation].length;
            int width = masks[rotation][0].length();
            result[rotation] = new SquareView[height][width];
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    if (masks[rotation][i].charAt(j) != ' ')
                        result[rotation][i][j] = new SquareView(type);
                }
            }
        }
        return result;
    }

    private static String[][][] rawData = new String[][][] {
        new String[][] {
            new String[] {
                "XX",
                "XX"
            }
        },
        new String[][] {
            new String[] {
                "X",
                "X",
                "X",
                "X"
            },
            new String[] {
                "XXXX"
            }
        },
        new String[][] {
            new String[] {
                "X ",
                "X ",
                "XX"
            },
            new String[] {
                "  X",
                "XXX"
            },
            new String[] {
                "XX",
                " X",
                " X"
            },
            new String[] {
                "XXX",
                "X  "
            }
        },
        new String[][] {
            new String[] {
                " X",
                " X",
                "XX"
            },
            new String[] {
                "XXX",
                "  X"
            },
            new String[] {
                "XX",
                "X ",
                "X "
            },
            new String[] {
                "X  ",
                "XXX"
            }
        },
        new String[][] {
            new String[] {
                " XX",
                "XX "
            },
            new String[] {
                "X ",
                "XX",
                " X"
            }
        },
        new String[][] {
            new String[] {
                "XX ",
                " XX"
            },
            new String[] {
                " X",
                "XX",
                "X "
            }
        },
        new String[][] {
            new String[] {
                "XXX",
                " X "
            },
            new String[] {
                "X ",
                "XX",
                "X "
            },
            new String[] {
                " X ",
                "XXX"
            },
            new String[] {
                " X",
                "XX",
                " X"
            }
        }
    };
}