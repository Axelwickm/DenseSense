import DenseSense.algorithms.Algorithm

import time
from collections import deque

import cv2
import numpy as np


class Tracker(DenseSense.algorithms.Algorithm.Algorithm):
    def __init__(self, dist_threshold=500, delete=0.3, hide=0.5,
                 min_frames=3, max_persistance_buffer_size=15, bbox_lag=0.4):
        super().__init__()
        self.frame = 0
        self.last_frame_time = None

        self.dist_threshold = dist_threshold
        self.delete = delete
        self.hide = hide
        self.min_frames = min_frames
        self.max_persistence_buffer_size = max_persistance_buffer_size
        self.bboxLag = bbox_lag

        self.old_people = []

        self.ghost_bounds = []
        self.kept_alive_index = []
        self.predictions = []

    def extract(self, people, time_now=None, debug=False):
        # Modifies the people list, making each person's id persistent,
        # and might add/remove entries from people for continuity's sake.
        self.frame += 1
        if time_now is None:
            time_now = time.time()
        if self.last_frame_time is None:
            delta_time = 0
        else:
            delta_time = self.last_frame_time - time_now
        self.last_frame_time = time_now

        # Make sure every person's original index is remembered
        for i in range(len(people)):
            people[i].attrs["originalIndex"] = i

        # Find optimal tracker <-> person configuration
        status = [-1 for _ in range(len(people))]  # xrange in python2
        associations = [(-1, 10000) for _ in range(len(self.old_people))]

        # Loop to link the new people vector to the old one
        i = 0
        while i < len(people):
            if status[i] != -1:
                i += 1
                continue
            person = people[i]
            ind, dist, displaced = self._match(person, associations)
            if ind is not None:
                associations[ind] = (i, dist)
                status[i] = ind
                if displaced is not None:
                    # A person (guaranteed lower index) has been displaced. Reconsider this person
                    status[displaced] = -1
                    i = displaced
                    continue
            i += 1

        self._update_existing_people(people, status, delta_time)
        self._change_states(people, associations)

    def _match(self, person, associations):
        # Convert to center and dim coords
        bounds = person.bounds
        dims = np.array([bounds[2] - bounds[0], bounds[3] - bounds[1]])
        center = np.array([bounds[0] + dims[0] / 2, bounds[1] + dims[1] / 2])

        # Find the closest viable tracked object
        displaced_person = None
        closest = float("inf")
        c_ind = -1
        for i in range(len(self.old_people)):  # Should be xrange in python2
            # Get distance between this person and the predicted center of the person
            predicted_center = np.squeeze(self.old_people[i].attrs["track"]["kalmanPrediciton"][:2])
            dist = np.linalg.norm(predicted_center - center)
            if dist < closest:
                # Is this the closest not reserved?
                if associations[i][0] == -1 or dist < associations[i][1]:
                    if associations[i][0] != -1:
                        displaced_person = associations[i][0]
                    else:
                        displaced_person = None
                    closest = dist
                    c_ind = i

        # If this was not close enough to be this object
        if self.dist_threshold < closest:
            return None, None, None

        return c_ind, closest, displaced_person

    def _update_existing_people(self, people, status, delta_time):
        for i in range(len(people)):
            person = people[i]
            bounds = person.bounds

            if status[i] == -1:  # If this is a new person
                dims = np.array([bounds[2] - bounds[0], bounds[3] - bounds[1]])
                center = np.array([bounds[0] + dims[0] / 2, bounds[1] + dims[1] / 2])

                # Construct a kalman filter
                t = cv2.CV_32F
                # FIXME: kalman magic numbers should be stored
                s = 6  # x, y, vx, vy, wx, w, h
                m = 4  # x, y, w, h
                kalman = cv2.KalmanFilter(s, m, 0, type=t)

                kalman.transitionMatrix = np.eye(s, dtype=np.float32)
                kalman.measurementMatrix = np.zeros((m, s), dtype=np.float32)
                kalman.measurementMatrix[0, 0] = 1
                kalman.measurementMatrix[1, 1] = 1
                kalman.measurementMatrix[2, 4] = 1
                kalman.measurementMatrix[3, 5] = 1

                kalman.processNoiseCov = 1e-2 * np.eye(s, dtype=np.float32)
                kalman.processNoiseCov[2, 2] = 5
                kalman.processNoiseCov[3, 3] = 5

                kalman.measurementNoiseCov = 1e-1 * np.eye(m, dtype=np.float32)

                kalman.controlMatrix = np.array(kalman.controlMatrix, dtype=np.float32)

                kalman.statePre = np.array([center[0], center[1], 0, 0, dims[0], dims[1]], dtype=np.float32)

                person.attrs["track"] = {
                    "lastSeenFrame": self.frame,
                    "lastSeenTime": self.last_frame_time,
                    "history": deque([1]),
                    "isVisible": False,
                    "kalman": kalman,
                    "kalmanPrediciton": kalman.statePre
                }
                self.old_people.append(person)

            else:  # Else, update this persons attributes
                ind = status[i]
                old_bounds = self.old_people[ind].bounds
                person = self.old_people[ind].become(person)
                people[i] = person
                self.old_people[ind] = person

                # Temporal bounds smoothing
                new_bounds = person.bounds
                updated_bounds = old_bounds * (1 - self.bboxLag) + new_bounds * self.bboxLag
                updated_bounds = updated_bounds.astype(np.int32)
                person.applyBounds(updated_bounds)

                bounds = updated_bounds
                dims = np.array([bounds[2] - bounds[0], bounds[3] - bounds[1]])
                center = np.array([updated_bounds[0] + dims[0] / 2, updated_bounds[1] + dims[1] / 2])

                # Update tracking attributes
                person.attrs["track"]["lastSeenFrame"] = self.frame
                person.attrs["track"]["lastSeenTime"] = self.last_frame_time
                person.attrs["track"]["history"].append(1)
                person.attrs["track"]["kalman"].transitionMatrix[0, 2] = delta_time * 0.4
                person.attrs["track"]["kalman"].transitionMatrix[1, 3] = delta_time * 0.4
                person.attrs["track"]["kalman"].correct(np.array(np.concatenate([center, dims]), dtype=np.float32))
                person.attrs["track"]["kalmanPrediction"] = person.attrs["track"]["kalman"].predict()

    def _change_states(self, people, associations):
        to_remove = []
        i = 0
        # Dynamically for every person
        while i < len(self.old_people):
            track = self.old_people[i].attrs["track"]

            # Add to history for all people who didn't show up this frame
            if i < len(associations):
                if associations[i][0] == -1:
                    track["history"].append(0)

            # Restrict size of history
            if self.max_persistence_buffer_size < len(track["history"]):
                track["history"].popleft()

            # Calculate average persistance over it's history
            persistence = float(sum(track["history"])) / len(track["history"])

            if self.hide < persistence and self.min_frames < len(track["history"]):
                track["isVisible"] = True  # The person is seen, and should keep being visible
            else:
                track["isVisible"] = False  # The person should not be seen
                if track["lastSeenFrame"] == self.frame:  # Suppress if he/she is
                    to_remove.append(self.old_people[i].attrs["originalIndex"])

            if persistence < self.delete:
                del self.old_people[i]  # Deleting from tracked objects
                continue

            # If shouldn't be removed and not seen, then hallucinate the person
            if track["lastSeenFrame"] != self.frame and track["isVisible"]:
                people.append(self.old_people[i])

            i += 1

        for index in sorted(to_remove, reverse=True):
            del people[index]

    def renderDebug(self, image, people):
        for person in self.old_people:
            bnds = person.bounds.astype(np.uint32)
            if not person.attrs["track"]["isVisible"]:
                # Draw dark rectangle to indicate this person is being suppressed
                image = cv2.rectangle(image, (bnds[0], bnds[1]),
                                      (bnds[2], bnds[3]), (100, 100, 100), 2)
            else:
                # Person is visible
                if person.attrs["track"]["lastSeenFrame"] != self.frame:
                    # Person is being hallucinated, indicated by filled rectangle
                    image[bnds[1]:bnds[3]:3, bnds[0]:bnds[2]:3] = person.color
                else:
                    # Draw bright rectangle
                    image = cv2.rectangle(image, (bnds[0], bnds[1]),
                                          (bnds[2], bnds[3]),
                                          person.color, 2)

        return image
