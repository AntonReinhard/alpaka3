# N-Body Simulation

This example simulates a number of particles by calculating pair-wise gravity between them. If PNGWriter is installed,
the results are saved to .png files in the executable's directory. You can use the following ffmpeg command to stitch
the images together into a video
file:

```bash 
$ ffmpeg -framerate 60 -i particles_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast particles.mkv
```
