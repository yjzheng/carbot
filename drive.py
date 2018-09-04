#!env python
#
# Auto-driving Bot
#
# Revision:      v1.2
# Released Date: Aug 20, 2018
#

from time import time
from PIL  import Image
from io   import BytesIO

#import datetime
import os
import cv2
import math
import numpy as np
import base64
import logging
import shutil

def logit(msg):
    print("%s" % msg)


class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0


    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error   = self._set_point - cur_value
        d_time  = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value   = error
            self._i_value   = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value   = d_error / d_time if d_time > 0 else 0.0
            self._output    = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time     = d_time
            self._d_error    = d_error
            self._last_time  = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point

    def get_p_value(self):
        return self._p_value

    def get_i_value(self):
        return self._i_value

    def get_d_value(self):
        return self._d_value

    def get_delta_time(self):
        return self._d_time

    def get_delta_error(self):
        return self._d_error

    def get_last_error(self):
        return self._last_error

    def get_last_time(self):
        return self._last_time

    def get_output(self):
        return self._output


class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)


    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
        from datetime import datetime
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0


    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi


    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def getBlackImg(img):
        lower = np.array([0, 0, 0])
        upper = np.array([15, 15, 15])
        return cv2.inRange(img, lower, upper)


    @staticmethod
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0/img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized


    @staticmethod
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((r, g, b))
        return flattened


    @staticmethod
    def _crop_image(img):
        bottom_half_ratios = (0.55, 1.0)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half


    @staticmethod
    def preprocess(img):
        img = ImageProcessor._crop_image(img)
        #img = ImageProcessor._normalize_brightness(img)
        img = ImageProcessor._flatten_rgb(img)
        return img


    @staticmethod
    def find_lines(img):
        grayed      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)
        #edged      = cv2.Canny(blurred, 0, 150)

        sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        sobel_abs_x = cv2.convertScaleAbs(sobel_x)
        sobel_abs_y = cv2.convertScaleAbs(sobel_y)
        edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

        lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
        return lines


    @staticmethod
    def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched = None, start_index = 0):
        if matched is not None:
            matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
            matched_angle = abs(np.pi/2 - matched_thetaB)

        for i in xrange(start_index, len(vectors)):
            distance, length, thetaA, thetaB, coord = vectors[i]

            if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
               (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):
                
                if matched is None:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

                heading_angle = abs(np.pi/2 - thetaB)

                if heading_angle > matched_angle:
                    continue
                if heading_angle < matched_angle:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if distance < matched_distance:
                    continue
                if distance > matched_distance:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if length < matched_length:
                    continue
                if length > matched_length:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

        return matched


    @staticmethod
    def find_steering_angle_by_line(img, last_steering_angle, debug = True):
        steering_angle_r = 0.0
        lines          = ImageProcessor.find_lines(img)

        if lines is None:
            return 0.0

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        camera_y     = image_height
        line_max_y   = image_height/2
        line_min_x   = image_width/4
        line_max_x   = image_width *0.75
        vectors      = []

        sum_slope=0
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1>line_max_y and y2 >line_max_y: # ignore line in the bottom
                    continue
                if x1>line_max_x or x2> line_max_x or x1 <line_min_x or x2<line_min_x:
                    continue

                if x1!=x2:
                    sum_slope= (y2-y1/x2-x1) +sum_slope
                    vectors.append((x1, y1, x2, y2))
                #todo: check what to do with straight line


                if debug:
                    # draw the wanted edges
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        #draw average slopline
        if len(vectors)>0:
            ave_slope = sum_slope/len(vectors)
            cv2.line(img, (camera_x,camera_y), (camera_x+10/ave_slope, camera_y+10), (255, 255, 0), 2)
        else:
            print "no line for average slop"
        ImageProcessor.show_image(img,"line")




        if (steering_angle_r - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
            last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

            if last:
                last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
                steering_angle_r = last_thetaB

                if debug:
                    #draw the last possible line
                    cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle_r))
            y = image_height    - int(r * math.sin(steering_angle_r))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f"
                  % (ImageProcessor.rad2deg(best_thetaA),
                     ImageProcessor.rad2deg(np.pi/2-steering_angle_r),
                     ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return ImageProcessor.rad2deg(np.pi/2 - steering_angle_r)


    @staticmethod
    def find_steering_angle_by_color(img, last_steering_angle, debug = True):
        b, g, r      = cv2.split(img)
        image_height = img.shape[0] #125

        image_width  = img.shape[1] #320
        camera_x     = image_width / 2 #160
        image_sample = slice(0, int(image_height * 0.1)) #get the top 1/5 image
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list   = [sr, sg, sb]
        tracks       = map(lambda x: len(x[x > 20]), [sr, sg, sb])#calc >20 element
        tracks_seen  = filter(lambda y: y > 50, tracks) #

        if len(tracks_seen) == 0:
            return 0.0
        wall  = ImageProcessor.getBlackImg(img)
        _wally, _wallx = np.where(wall == 255)


        if len(_wallx[_wallx > 20]) >0:
            wallpx = np.mean(_wallx)
            cv2.line(wall, (camera_x, image_height), (int(wallpx), 0), (255, 0, 255), 2)
        ImageProcessor.show_image(wall, "wall")


        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        #maximum_color_idx =1
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)
        px = np.mean(_x)
        steering_angle = math.atan2(image_height, (px - camera_x))
        angle= 90-ImageProcessor.rad2deg(steering_angle)
        ImageProcessor.show_image(_target, "_target")
        cv2.line(img, (camera_x, image_height), (int(px), 0), (255, 0, 255), 2)
        ImageProcessor.show_image(img, "img")

        return angle * 2.0


class AutoDrive(object):
    STEERING_PID_Kp             = 0.3
    STEERING_PID_Ki             = 0.01
    STEERING_PID_Kd             = 0.1
    STEERING_PID_max_integral   = 10
    THROTTLE_PID_Kp             = 0.02
    THROTTLE_PID_Ki             = 0.005
    THROTTLE_PID_Kd             = 0.02
    THROTTLE_PID_max_integral   = 0.5
    MAX_STEERING_HISTORY        = 3
    MAX_THROTTLE_HISTORY        = 3
    DEFAULT_SPEED               = 1.0

    debug = True

    def __init__(self, car, record_folder = None):
        self._record_folder    = record_folder
        self._steering_pid     = PID(Kp=self.STEERING_PID_Kp  , Ki=self.STEERING_PID_Ki  , Kd=self.STEERING_PID_Kd  , max_integral=self.STEERING_PID_max_integral  )
        self._throttle_pid     = PID(Kp=self.THROTTLE_PID_Kp  , Ki=self.THROTTLE_PID_Ki  , Kd=self.THROTTLE_PID_Kd  , max_integral=self.THROTTLE_PID_max_integral  )
        self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)
        self._steering_history = []
        self._throttle_history = []
        self._car = car
        self._car.register(self)


    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        track_img     = ImageProcessor.preprocess(src_img)
        #current_angle = ImageProcessor.find_steering_angle_by_color(track_img, last_steering_angle, debug = True)
        current_angle = ImageProcessor.find_steering_angle_by_line(track_img, last_steering_angle, debug = self.debug)
        steering_angle = self._steering_pid.update(-current_angle)
        throttle       = self._throttle_pid.update(speed)

        if self.debug:
            ImageProcessor.show_image(src_img, "source")
            ImageProcessor.show_image(track_img, "track")
            #logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
            #logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
            #logit("info: %s" % repr(info))

        if self._record_folder:
            suffix = "-deg%0.3f" % (ImageProcessor.rad2deg(steering_angle))
            #ImageProcessor.save_image("bot_log", src_img  , prefix = "cam", suffix = suffix)
            ImageProcessor.save_image("bot_log", track_img, prefix = "trk", suffix = suffix)
        #smooth the control signals
        self._steering_history.append(steering_angle)
        self._steering_history = self._steering_history[-self.MAX_STEERING_HISTORY:]
        self._throttle_history.append(throttle)
        self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]

        self._car.control(sum(self._steering_history)/self.MAX_STEERING_HISTORY, sum(self._throttle_history)/self.MAX_THROTTLE_HISTORY)


class Car(object):
    MAX_STEERING_ANGLE = 40.0


    def __init__(self, control_function):
        self._driver = None
        self._control_function = control_function


    def register(self, driver):
        self._driver = driver


    def on_telemetry(self, dashboard):
        #normalize the units of all parameters
        #last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        last_steering_angle = float(dashboard["steering_angle"])
        throttle            = float(dashboard["throttle"])
        brake               = float(dashboard["brakes"])
        speed               = float(dashboard["speed"])
        img                 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))
        del dashboard["image"]
        print datetime.now(), dashboard;
        total_time = float(dashboard["time"])
        elapsed    = total_time

        #if elapsed >20:
            #print "elapsed: " +str(elapsed)
            #send_restart()


        info = {
            "lap"    : int(dashboard["lap"]) if "lap" in dashboard else 0,
            "elapsed": elapsed,
            "status" : int(dashboard["status"]) if "status" in dashboard else 0,
        }
        self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)


    def control(self, steering_angle, throttle):
        #convert the values with proper units
        #steering_angle = min(max(ImageProcessor.rad2deg(steering_angle), -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)


if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()

    if args.record:
        if os.path.exists(args.record):
            shutil.rmtree(args.record)
        os.makedirs(args.record)

        logit("Start recording images to %s..." % args.record)

    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)
        print "send control, steering angle: " + str(steering_angle) + " throttle: " +str (throttle)
    def send_restart():
        sio.emit(
            "restart",
            data={},
            skip_sid=True)

    car = Car(control_function = send_control)
    drive = AutoDrive(car, args.record)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_telemetry(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        send_restart()
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# vim: set sw=4 ts=4 et :

