import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Comparator;

public class Pixelify {

    static class Pair {
        int s,t;
        float cost;
        Pair(int s,int t,float cost){this.s=s;this.t=t;this.cost=cost;}
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java Pixelify source.jpg target.jpg out.png [proximity]");
            return;
        }

        String source = args[0];
        String target = args[1];
        String outPath = args[2];  
        float proximity = args.length >= 4 ? Float.parseFloat(args[3]) : 0.5f;

        BufferedImage src = ImageIO.read(new File(source));
        BufferedImage tgtOrig = ImageIO.read(new File(target));
        BufferedImage tgt = resize(tgtOrig, src.getWidth(), src.getHeight()); 

        BufferedImage outImg = Pixeli(src, tgt, proximity);

        ImageIO.write(outImg, "PNG", new File(outPath));

        System.out.println("Saved: " + outPath);
    }

    static BufferedImage resize(BufferedImage img, int width, int height) {
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return dimg;
    }

    static BufferedImage Pixeli(BufferedImage src, BufferedImage tgt, float proximity) {
        int width = src.getWidth();
        int height = src.getHeight();
        int N = width * height;

        float[][] srcColor = new float[N][3];
        float[][] tgtColor = new float[N][3];
        float[][] srcCenter = new float[N][2];
        float[][] tgtCenter = new float[N][2];

        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgbS = src.getRGB(x, y);
                srcColor[idx] = rgbToArr(rgbS);
                srcCenter[idx][0] = (float)x / (width-1);
                srcCenter[idx][1] = (float)y / (height-1);

                int rgbT = tgt.getRGB(x, y);
                tgtColor[idx] = rgbToArr(rgbT);
                tgtCenter[idx][0] = (float)x / (width-1);
                tgtCenter[idx][1] = (float)y / (height-1);

                idx++;
            }
        }

        PriorityQueue<Pair> pq = new PriorityQueue<>(Comparator.comparingDouble(p -> p.cost));
        for (int s = 0; s < N; s++) {
            for (int t = 0; t < N; t++) {
                float cc = colorDistSq(srcColor[s], tgtColor[t]) / (255f*255f*3f);
                float sc = spatialDistSq(srcCenter[s], tgtCenter[t]) / 2f;
                float cost = cc * 1.0f + sc * proximity;
                pq.add(new Pair(s,t,cost));
            }
        }

        boolean[] assignedS = new boolean[N];
        boolean[] assignedT = new boolean[N];
        int[] srcToTgt = new int[N];
        Arrays.fill(srcToTgt, -1);
        int assigned = 0;
        while (!pq.isEmpty() && assigned < N) {
            Pair p = pq.poll();
            if (!assignedS[p.s] && !assignedT[p.t]) {
                assignedS[p.s] = true;
                assignedT[p.t] = true;
                srcToTgt[p.s] = p.t;
                assigned++;
            }
        }

        BufferedImage outFull = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int s2 = 0; s2 < N; s2++) {
            int t = srcToTgt[s2];
            int xs = s2 % width;
            int ys = s2 / width;
            int rgb;
            if (t >= 0) {
                float[] c = tgtColor[t];
                rgb = arrToRgb(c);
            } else {
                rgb = arrToRgb(srcColor[s2]);
            }
            outFull.setRGB(xs, ys, rgb);
        }

        return outFull;
    }

    static float[] rgbToArr(int rgb){
        int r = (rgb>>16)&0xFF;
        int g = (rgb>>8)&0xFF;
        int b = rgb&0xFF;
        return new float[]{r,g,b};
    }
    static int arrToRgb(float[] c){
        int r = clamp(Math.round(c[0]));
        int g = clamp(Math.round(c[1]));
        int b = clamp(Math.round(c[2]));
        return (r<<16)|(g<<8)|b;
    }
    static int clamp(int v){ if (v<0) return 0; if (v>255) return 255; return v;}
    static float colorDistSq(float[] a, float[] b){
        float dx = a[0]-b[0];
        float dy = a[1]-b[1];
        float dz = a[2]-b[2];
        return dx*dx + dy*dy + dz*dz;
    }
    static float spatialDistSq(float[] a, float[] b){
        float dx = a[0]-b[0];
        float dy = a[1]-b[1];
        return dx*dx + dy*dy;
    }
}
