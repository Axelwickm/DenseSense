from queue import Queue
from threading import Thread, Event
import time

import pafy
import ffmpeg
import numpy as np



class YoutubeLoader:
    def __init__(self, chunk_buffer_size=10, dimensions=(400, 300), verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Starting YoutubeLoader")

        self.chunk_buffer_max_size = chunk_buffer_size
        self.chunk_max_size = 128

        self.goal_dimensions = np.array(dimensions)

        self.current_buffer_size = 0
        self.chunk_buffer = {}

        self.video_queue = []

        self.video_cursor = None
        self.chunk_cursor = None
        self.frame_cursor = None

        self.downloader = Thread(target=self._download)
        self.downloader.start()

    def queue_video(self, key, start_time, end_time, fps_mean=5, fps_std=2):
        if self.verbose:
            print("Queueing video: {}, {}->{}".format(key, start_time, end_time))
        self.video_queue.append((key, start_time, end_time, fps_mean, fps_std))
        self.chunk_buffer[key] = [[Event(), ()]]

    def get(self, video_index, frame_index):
        self.video_cursor = video_index
        self.chunk_cursor = int(np.floor(frame_index / self.chunk_max_size))
        self.frame_cursor = frame_index % self.chunk_max_size

        # Get data (wait for downloader if not already done)
        key, start_time, end_time, _, _ = self.video_queue[self.video_cursor]
        while self.chunk_cursor >= len(self.chunk_buffer[key]):
            if self.verbose:
                print("Waiting for download thread to start with video...")
            time.sleep(0.1)

        self.chunk_buffer[key][self.chunk_cursor][0].wait()
        frames, times, indices, is_last_chunk = self.chunk_buffer[key][self.chunk_cursor][1]

        # Format return output
        if frames is None:
            is_last_frame = True
            out = None, None, True
        else:
            is_last_frame = is_last_chunk and self.frame_cursor == len(frames) - 1
            out = frames[self.frame_cursor], times[self.frame_cursor], is_last_frame

        if self.verbose and False:
            print("Yield video {} ({}), chunk {}, frame {}:, last frame {}"
                  .format(key, self.video_cursor, self.chunk_cursor, self.frame_cursor, is_last_frame))

        # Do increments (semi-redundant)
        if frames is None:
            pass
        elif self.frame_cursor == len(frames) - 1:
            self.frame_cursor += 1

        if is_last_frame:
            self.video_cursor = (self.video_cursor+1) % len(self.video_queue)
            self.frame_cursor = 0
            self.chunk_cursor = 0

        return out

    def frames(self):
        for i in range(len(self.video_queue)):
            frame_index = 0
            while not frame_index == -1:
                out = self.get(i, frame_index)
                frame_index = -2 if out[2] else frame_index
                frame_index += 1
                if out[0] is None:
                    break
                yield out

    def _download(self):
        currently_downloaded_videos = []

        while True:
            if len(self.video_queue) == 0 or self.video_cursor is None:
                continue

            # Find what video to download
            download_video_cursor = self.video_cursor
            if len(currently_downloaded_videos) != 0:
                while download_video_cursor in currently_downloaded_videos:
                    download_video_cursor = (download_video_cursor + 1) % len(self.video_queue)

            # Get video queue info
            key, start_time, end_time, fps_mean, fps_std = self.video_queue[download_video_cursor]
            currently_downloaded_videos.append(download_video_cursor)
            if self.verbose:
                print("\tDownloading video {}, {}->{}".format(download_video_cursor, start_time, end_time))

            # Download video info
            try:
                video = pafy.new(key)
            except IOError as e:
                if self.verbose:
                    print("\tVideo probably doesn't exists anymore (or you don't have internet)")
                self.chunk_buffer[key] = [[Event(), (None, None, None, True)]]
                self.chunk_buffer[key][0][0].set()
                continue

            stream = self._find_most_fitting_stream(video)
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

            while chunk_start < end_time and 0 < len(frame_times):
                end_frame = min(self.chunk_max_size, len(frame_times)-1)
                chunk_end = frame_times[end_frame]
                chunk_duration = chunk_end - chunk_start

                if self.verbose:
                    print("\tDownloading video {} chunk {}: {}s -> {}s".format(download_video_cursor,
                                                                               len(self.chunk_buffer[key])-1,
                                                                               chunk_start, chunk_end))

                # Download chunk
                video = ffmpeg.input(stream.url, ss=chunk_start, t=chunk_duration,
                                     format="mp4", loglevel="error")
                video = video.output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="error")
                out, err = video.run(capture_stdout=True)
                downloaded_frames = np.frombuffer(out, np.uint8).reshape([-1, dimensions[1], dimensions[0], 3])

                # Select the right fps and save to array
                fps = downloaded_frames.shape[0] / chunk_duration
                current_frames = frame_times[:end_frame + 1]
                frame_times = frame_times[end_frame + 1:]
                frames_inds = np.floor((current_frames - chunk_start) * fps * 0.999).astype(np.int32)

                different_from_previous = np.diff(frames_inds, prepend=[1]) != 0
                frames_inds = frames_inds[different_from_previous]
                current_frames = current_frames[different_from_previous]
                filtered_frames = downloaded_frames[frames_inds]

                # Update loop
                chunk_start += chunk_duration

                # Pass on data
                last_chunk = len(frame_times) == 0
                self.chunk_buffer[key][-1][1] = filtered_frames, current_frames, frames_inds, last_chunk
                self.chunk_buffer[key].append([Event(), ()])
                self.chunk_buffer[key][-2][0].set()
                self.current_buffer_size += 1
                if self.verbose:
                    print("\tWas last:", last_chunk)
                    print("\tTotal chunks now buffered: "+str(self.current_buffer_size)
                          + " (max "+str(self.chunk_buffer_max_size)+")")

                while True:
                    def release_chunk(video_i, chunk_i):
                        k = self.video_queue[video_i][0]
                        if self.chunk_buffer[k][chunk_i][0].isSet():
                            self.current_buffer_size -= 1
                        self.chunk_buffer[k][chunk_i][1] = None

                    def should_release(video_i):
                        delta = video_i - self.video_cursor
                        if delta < 0:
                            delta += len(self.video_queue)
                        allowed = 0 <= delta <= self.chunk_buffer_max_size
                        return not allowed

                    # Release old chunks of current video
                    playing_key = self.video_queue[self.video_cursor][0]
                    for i in range(len(self.chunk_buffer[playing_key])):
                        if i < self.chunk_cursor:
                            release_chunk(self.video_cursor, i)

                    # Release old videos
                    release_videos = list(filter(should_release, currently_downloaded_videos))

                    for rv in release_videos:
                        if self.verbose:
                            print("\tRelease video: " + str(rv))
                        currently_downloaded_videos.remove(rv)
                        current_key = self.video_queue[rv][0]
                        for i, rc in enumerate(self.chunk_buffer[current_key]):
                            release_chunk(rv, i)
                        self.chunk_buffer[current_key] = [[Event(), ()]]

                    if self.chunk_buffer_max_size <= self.current_buffer_size:
                        if self.verbose:
                            print("\tWaiting to download next chunk")
                        time.sleep(0.1)
                    else:
                        break

            if self.verbose:
                print("\tFinished downloading", key, "\n")

    def _find_most_fitting_stream(self, video):
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

    yl = YoutubeLoader(chunk_buffer_size=6, verbose=False)
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
    yl.queue_video("iKygSW4Xpjc", 1, 19*60)
    yl.queue_video("-wPTadJB5As", 5, 20)

    while True:
        for frame, t, last in yl.frames():
            cv2.imshow("frame", frame)
            cv2.waitKey(20)



