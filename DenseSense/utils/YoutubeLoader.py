import pafy
import ffmpeg
import numpy as np


class YoutubeLoader:
    def __init__(self, chunk_buffer_size=200, dimensions=(400, 300),
                 verbose=False, save_to_lmdb=False):
        print("Starting YoutubeLoader")
        self.chunk_buffer_size = chunk_buffer_size
        self.goal_dimensions = np.array(dimensions)
        self.verbose = verbose
        self.save_to_lmdb = save_to_lmdb

        self.video_queue = []
        self.video_cursor = 0
        self.chunk_time = 15

    def queue_video(self, key, start_time, end_time, fps_mean=5, fps_std=2):
        if self.verbose:
            print("Queuing video", key, start_time, end_time)

        try:
            video = pafy.new(key)
        except IOError as e:
            print("Video probably doesn't exists")
            raise e

        stream = self._findMostFittingStream(video)
        dimensions = stream.dimensions

        # What frames to store
        duration = end_time - start_time
        frame_times = np.random.normal(1/fps_mean, 1/(fps_mean+fps_std), fps_mean*duration*2)
        frame_times = np.cumsum(frame_times)
        last_frame = np.argmax(duration+10000 < frame_times)
        last_frame = len(frame_times) if last_frame == 0 else last_frame
        frame_times = frame_times[:last_frame]
        downloaded_video = np.array((len(frame_times), dimensions[1], dimensions[0], 3), dtype=np.uint8)
        print("Frames to download: {}".format(len(frame_times)))

        chunk_start = start_time
        next_frame = 0
        remaining_frame_times = frame_times.copy()
        while chunk_start < start_time + duration:
            # Download chunk
            chunk_end = min(chunk_start + self.chunk_time, end_time)
            chunk_duration = chunk_end - chunk_start
            video = ffmpeg.input(stream.url, ss=chunk_start, t=chunk_duration, format="mp4", loglevel="error")
            video = video.output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="error")
            out, err = video.run(capture_stdout=True)
            downloaded_frames = np.frombuffer(out, np.uint8).reshape([-1, dimensions[1], dimensions[0], 3])

            # Select the right fps and save to array
            fps = downloaded_frames.shape[0]/chunk_duration
            endFrame = np.argmax(chunk_end < remaining_frame_times)
            endFrame = len(remaining_frame_times) if endFrame == 0 else endFrame
            frames_inds = np.floor((remaining_frame_times[:endFrame]-remaining_frame_times[0])*fps).astype(np.int32)
            print("end frame", endFrame)
            print(len(frames_inds))
            print(frames_inds)
            print(frames_inds+next_frame)
            downloaded_video[next_frame+frames_inds] = downloaded_frames[frames_inds, :, :, 3]
            print("selected frames", downloaded_frames[frames_inds].shape)
            chunk_start += self.chunk_time
            remaining_frame_times = remaining_frame_times[endFrame:]


        print("Finished downloading", key)
        #print(downloaded_video.shape)
        return downloaded_video

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

    def getNextAndBuffer(self, buffer):
        print("Video buffer keys", self.videoBuffer.keys())
        print("Upcoming", self.upcoming)
        s = ""
        for i in self.upcoming:
            s += self.ava[i[0]]["key"] + ", "
        print("Upcoming keys: " + s)
        if self.lastVideo is not None:
            lastKey = self.ava[self.lastVideo[0]]["key"]
            for i, j in self.upcoming:
                if lastKey == self.ava[i]["key"]:
                    break
            else:
                del self.videoBuffer[lastKey]
                print("------------- UNLOAD VIDEO -- " + lastKey + " --- " + str(
                    len(self.videoBuffer)) + " videos left ---")

        # What classes should be choosen
        whatClasses = np.random.choice(self.actions.keys(), buffer - len(self.upcoming))

        # Move each cursors of chosen classes to next positions,
        # and queue all video downloads
        for c in whatClasses:
            cursor = self.classCursors[c]
            key = self.ava[cursor[0]]["key"]
            starttime = self.ava[cursor[0]]["startTime"]
            endtime = self.ava[cursor[0]]["endTime"]
            self.upcoming.append(cursor[:])
            if key not in self.videoBuffer:
                self.videoBuffer[key] = [threading.Event(), None]
                threading.Thread(target=self.download_video, args=(key, starttime, endtime)).start()

            while True:
                cursor[1] += 1
                if len(self.ava[cursor[0]]["people"]) == cursor[1]:
                    cursor[0] = (cursor[0] + 1) % len(self.ava)
                    cursor[1] = 0
                person = self.ava[cursor[0]]["people"][cursor[1]]
                if str(c) in person.keys():
                    self.classCursors[c] = cursor
                    break

        # Unload video if it's not appearing again for a while
        current = self.upcoming[0]
        self.upcoming = self.upcoming[1:]

        currentKey = self.ava[current[0]]["key"]
        if self.videoBuffer[currentKey][1] is None:
            self.videoBuffer[currentKey][0].wait()  # Wait for video to have been downloaded

        if self.videoBuffer[currentKey][1] is False:
            print("Video " + currentKey + " has been removed. Skipping...")
            self.lastVideo = None
            del self.videoBuffer[currentKey]  # This video has been removed from YT
            i = 0
            while i < len(self.upcoming):
                ind = self.upcoming[i][0]
                if currentKey == self.ava[i]["key"]:
                    del self.upcoming[ind]
                else:
                    i += 1

            return self.getNextAndBuffer(buffer)

        if tuple(current) in self.recent:
            print("Cursor " + str(current) + " was recently processed. Skipping...")
            self.lastVideo = None
            return self.getNextAndBuffer(buffer)

        self.recent.append(tuple(current))
        if 15 < len(self.recent):
            self.recent.popleft()
        self.lastVideo = current
        return current, currentKey


if __name__ == "__main__":
    import cv2

    yl = YoutubeLoader(verbose=True)
    vid = yl.queue_video("dVPqWh39HJ0", 0, 123)

    for index, frame in enumerate(vid):
        print("Showing frame: "+str(index))
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
