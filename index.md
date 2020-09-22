---
summary: False
---
# Verae data science cheat-book
I'm Vera, and this is my cheat-book. If you want correct, complete or add something, don't hesitate to do a pull request. Check out a [smaller but uber-useful cheat sheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning) I found while writing this.

## Docker

```bash
docker ps
docker exec -it [NAME] bash
docker attach -i [NAME]
docker build
docker-compose up
```

## Cookiecutter
[dvc and mlflow data science cookiecutter](https://github.com/iKintosh/cookiecutter-data-science)

## Business
[Why we have to test](https://dealbook.nytimes.com/2012/08/02/knight-capital-says-trading-mishap-cost-it-440-million/)
Why do so many incompetent men become leaders?

On leadership: [video](https://www.youtube.com/watch?v=zeAEFEXvcBg) | [book](https://www.goodreads.com/en/book/show/41959331-why-do-so-many-incompetent-men-become-leaders)
## Maker

[raspberry HD video recording](https://www.arrow.com/en/research-and-events/articles/pi-bandwidth-with-video)

[arduino interrupts](https://learn.adafruit.com/multi-tasking-the-arduino-part-2/timers)

## SSH

### Close stuck sessions
`~.` to terminate the connection

## Create a desktop capable ubuntu 20 on the cloud (with GPU)
[web tutorial](http://leadtosilverlining.blogspot.com/2019/01/setup-desktop-environment-on-google.html)

### bash notes
````bash
# Let´´ connect to the machine
ssh -i ~/.ssh/odd -L 5901:localhost:5901 vera_odd_co@34.83.165.112

# Download nvidia drivers
curl 'http://us.download.nvidia.com/tesla/440.64.00/NVIDIA-Linux-x86_64-440.64.00.run' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:76.0) Gecko/20100101 Firefox/76.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en,en-US;q=0.7,es;q=0.3' --compressed -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Cookie: vid=dc0c1504-ba63-42cf-95c7-eba68721fd39' -H 'Upgrade-Insecure-Requests: 1' > NVIDIA-Linux-x86_64-440.64.00.run

chmod +x NVIDIA-Linux-x86_64-440.64.00.run

sudo ./NVIDIA-Linux-x86_64-440.64.00.run
````


## Misc
### Streaming
[from ffmpeg to python](https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#jupyter-stream-editor)

[ffmpeg streaming docs](https://trac.ffmpeg.org/wiki/StreamingGuide)

[ffserver](https://trac.ffmpeg.org/wiki/ffserver)

[obs and ffmpeg streaming](https://obsproject.com/forum/resources/obs-studio-send-an-udp-stream-to-a-second-pc-using-obs.455/)

### Other
[Voice Activity Detector](https://github.com/wiseman/py-webrtcvad)

[pyro tutorial](https://mltrain.cc/events/mltrain-uai-2018/)

[last models](https://paperswithcode.com/)