from queue import Queue
from threading import Thread, Event

import pafy
import ffmpeg
import numpy as np

import time


class YoutubeLoader:
    def __init__(self, chunk_buffer_size=5, dimensions=(400, 300), verbose=False):
        print("Starting YoutubeLoader")
        self.chunk_buffer_max_size = chunk_buffer_size
        self.goal_dimensions = np.array(dimensions)
        self.verbose = verbose

        self.current_buffer_size = 0
        self.chunk_buffer = {}
        self.download_next_chunk = Event()

        self.video_queue = []
        self.video_cursor = 0
        self.chunk_time = 15

        self.downloader = Thread(target=self._download)
        self.downloader.start()

    def queue_video(self, key, start_time, end_time, fps_mean=5, fps_std=2):
        if self.verbose:
            print("Queueing video: {}, {}->{}".format(key, start_time, end_time))
        self.video_queue.append((key, start_time, end_time, fps_mean, fps_std))

    def frames(self):
        while True:
            if self.verbose:
                print("Read new video {}".format(self.video_cursor))

            key, start_time, end_time, _, _ = self.video_queue[self.video_cursor]
            while key not in self.chunk_buffer:
                time.sleep(0.05)
                if self.verbose:
                    print("Blocking main wait")

            lastChunk = False
            chunkCursor = 0
            while not lastChunk:
                if self.verbose:
                    print("Reading video {}, chunk {}".format(self.video_cursor, chunkCursor))
                    print("mmmmmmmmmm")
                    print("key", key, "chunkCursor", chunkCursor)
                    print("cbkey", self.chunk_buffer[key])
                    print("cbkeycc", self.chunk_buffer[key][chunkCursor])
                    print("babababa\n")
                self.chunk_buffer[key][chunkCursor][0].wait()
                frames, times, inds, lastChunk = self.chunk_buffer[key][chunkCursor][1]
                if frames is None:
                    lastChunk = True
                    continue

                # Release chunk
                self.chunk_buffer[key][chunkCursor] = None
                self.current_buffer_size -= 1
                self.download_next_chunk.set()
                chunkCursor += 1

                for frameIndex, frame in enumerate(frames):
                    yield frame, key, self.video_cursor, frameIndex, times[frameIndex]

            self.video_cursor = (self.video_cursor+1) % len(self.video_queue)
            del self.chunk_buffer[key]

    def _download(self):
        download_video_cursor = 0
        while True:
            if len(self.video_queue) == 0:
                continue

            key, start_time, end_time, fps_mean, fps_std = self.video_queue[download_video_cursor]
            if self.verbose:
                print("\tDownloading video {}, {}->{}".format(key, start_time, end_time))

            # FIXME: don't download again if video already in buffer

            try:
                video = pafy.new(key)
            except IOError as e:
                if self.verbose:
                    print("Video probably doesn't exists anymore (or you don't have internet)")
                download_video_cursor += 1
                self.chunk_buffer[key] = [[Event(), (None, None, None, True)]]
                self.chunk_buffer[key][0][0].set()
                continue

            stream = self._findMostFittingStream(video)
            dimensions = stream.dimensions

            # What frames to store
            duration = end_time - start_time
            hz_mean = 1.0 / fps_mean
            hz_std = hz_mean - 1.0 / (fps_mean + fps_std)
            frame_times = np.random.normal(hz_mean, hz_std, int(fps_mean * duration * 2))
            frame_times = start_time+np.sort(np.cumsum(frame_times))
            first_frame = np.argmax(start_time < frame_times)
            last_frame = np.argmax(end_time < frame_times)
            last_frame = len(frame_times) if last_frame == 0 else last_frame
            frame_times = frame_times[first_frame:last_frame]

            chunk_start = start_time
            remaining_frame_times = frame_times.copy()

            self.chunk_buffer[key] = [[Event(), ()]]

            while chunk_start < end_time:
                # Get chunk length
                chunk_end = min(chunk_start + self.chunk_time, end_time)
                chunk_duration = chunk_end - chunk_start

                if self.verbose:
                    print("\tDownloading video {} chunk {}: {}s -> {}s".format(download_video_cursor,
                                                                               len(self.chunk_buffer[key])-1,
                                                                               chunk_start, chunk_end))

                # Download chunk
                video = ffmpeg.input(stream.url, ss=chunk_start, t=chunk_duration, format="mp4", loglevel="error")
                video = video.output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="error")
                out, err = video.run(capture_stdout=True)
                downloaded_frames = np.frombuffer(out, np.uint8).reshape([-1, dimensions[1], dimensions[0], 3])

                # Select the right fps and save to array
                fps = downloaded_frames.shape[0] / chunk_duration
                endFrame = np.argmax(chunk_end < remaining_frame_times)
                endFrame = len(remaining_frame_times) if endFrame == 0 else endFrame

                current_frames = remaining_frame_times[:endFrame]
                remaining_frame_times = remaining_frame_times[endFrame:]
                frames_inds = np.floor((current_frames - chunk_start) * fps).astype(np.int32)

                differentFromPrevious = np.diff(frames_inds, prepend=[1]) != 0
                frames_inds = frames_inds[differentFromPrevious]
                current_frames = current_frames[differentFromPrevious]

                filteredFrames = downloaded_frames[frames_inds]

                # Update loop
                chunk_start += chunk_duration

                # Pass on data
                lastChunk = end_time <= chunk_start
                self.chunk_buffer[key][-1][1] = filteredFrames, current_frames, frames_inds, lastChunk
                self.chunk_buffer[key].append([Event(), ()])
                self.chunk_buffer[key][-2][0].set()
                self.current_buffer_size += 1
                if self.verbose:
                    print("\tTotal chunks now buffered: "+str(self.current_buffer_size))
                if self.chunk_buffer_max_size <= self.current_buffer_size:
                    if self.verbose:
                        print("\tWaiting to download next chunk")
                    self.download_next_chunk.clear()
                    self.download_next_chunk.wait()

            if self.verbose:
                print("\tFinished downloading", key)
            download_video_cursor = (download_video_cursor+1) % len(self.video_queue)

    def _findMostFittingStream(self, video):
        bestStream = None
        bestMatch = float("inf")
        for stream in video.streams:
            if stream.extension != "mp4":
                continue
            dimensions = np.array(stream.dimensions)
            distance = np.linalg.norm(np.abs(self.goal_dimensions-dimensions))
            if distance < bestMatch:
                bestMatch = distance
                bestStream = stream
        return bestStream


if __name__ == "__main__":
    import cv2

    yl = YoutubeLoader(chunk_buffer_size=15, verbose=True)
    yl.queue_video("dVPqWh39HJ0", 0, 10)
    yl.queue_video("IzIBpFDRr5g", 20, 30)
    yl.queue_video("DSYXObynLWY", 30, 70)
    yl.queue_video("Idontexiste", 30, 70)
    yl.queue_video("UnETVMI4tY8", 0, 10)
    yl.queue_video("3DBQxxvuWYA", 10, 11)
    yl.queue_video("ERg3JvmnkUI", 0.02, 30)
    yl.queue_video("Ze25iPU5f-M", 0.02, 0.5)
    yl.queue_video("hzd53i7hhhA", 0, 70)
    yl.queue_video("bpm-YTucLrA", 5, 20)
    yl.queue_video("vc8MddDFRw4", 0, 20)
    yl.queue_video("-wPTadJB5As", 5, 20)

    for frame, videoKey, videoIndex, frameIndex, frameTime in yl.frames():
        cv2.imshow("frame", frame)
        cv2.waitKey(30)


