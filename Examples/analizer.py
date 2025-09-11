import cv2
import laserbeamsize as lbs
import matplotlib.pyplot as plt
import numpy as np

from queue import Queue
from vmbpy import *

CAMERA_ID = "DEV_000F31F42C02"
FORMAT = PixelFormat.Mono8
ENTER_KEY_CODE = 13
INT_TIME = 7000
PIXEL_SIZE = 15

def print_camera(cam:Camera):
    print('/// Camera Name   : {}'.format(cam.get_name()))
    print('/// Model Name    : {}'.format(cam.get_model()))
    print('/// Camera ID     : {}'.format(cam.get_id()))
    print('/// Serial Number : {}'.format(cam.get_serial()))
    print('/// Interface ID  : {}\n'.format(cam.get_interface_id()))

def setup_camera(cam: Camera):
    with cam:

        try:
            cam.ExposureAuto.set('Off')
            cam.ExposureTime.set(INT_TIME)
        except (AttributeError, VmbFeatureError):
            pass

        try:
            cam.BalanceWhiteAuto.set('Continuous')
        except (AttributeError, VmbFeatureError):
            pass

        try:
            cam.set_pixel_format(FORMAT)
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VmbFeatureError):
            pass

def integrated_intensity(image, background = True):
    """
    Return the integrated intensity of the beam as the sum of the pixel values of 

    Args:
        image: 2D array of image with beam spot.
        background (optional): if true, the ISO 11146 background is subtracted from the image.
    Returns:
        x: Pixel of the center of the beam in the horizontal axis.
        y: Pixel of the center of the beam in the verticalal axis.
        d_major: Length of the major axis of the beam ellipse in pixels.
        d_minor: Length of the major axis of the beam ellipse in pixels.
        phi: Counterclockwise angle in radians between the major axis and the
             horizontal axis.
        intensity: Integrated intensity of the beam.
    """

    if background:
        image = lbs.subtract_iso_background(image)

    pixels_x = np.array([],dtype=int)
    pixels_y = np.array([],dtype=int)

    x, y, d_major, d_minor, phi = lbs.beam_size(image)
    ellipse_x, ellipse_y = lbs.ellipse_arrays(x, y, d_major, d_minor, phi,npoints=400)

    for ell_x, ell_y in zip (ellipse_x,ellipse_y):
        curr_x, curr_y = lbs.image_tools.line(x,y,ell_x, ell_y)
        pixels_x = np.concatenate((pixels_x, curr_x))
        pixels_y = np.concatenate((pixels_y, curr_y))

    points = list(zip(pixels_x, pixels_y))
    unique_coordinates = list(set(points))
    intensity = 0
    for aux_x, aux_y in unique_coordinates:
        try:
            intensity += image[aux_x, aux_y]
        except IndexError:
            pass
    intensity /= INT_TIME
    return x, y, d_major, d_minor, phi, intensity

class Handler:
    def __init__(self):
        self.display_queue = Queue(10)

    def get_image(self):
        return self.display_queue.get(True)
    
    def get_info(self):
        return self.x, self.y, self.major, self.minor, self.phi, self.intensity

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            beam = frame.as_numpy_ndarray()
            beam = beam[:, :, 0]
            self.x, self.y, self.major, self.minor, self.phi, self.intensity = integrated_intensity(beam)
            self.major *= PIXEL_SIZE
            self.minor *= PIXEL_SIZE
            if frame.get_pixel_format() == FORMAT:
                display = frame
            else:
                display = frame.convert_pixel_format(FORMAT)

            self.display_queue.put(display.as_opencv_image(), True)

        cam.queue_frame(frame)

def main():
    with VmbSystem.get_instance() as vmb:
        with vmb.get_camera_by_id(CAMERA_ID) as cam:
            setup_camera(cam)
            handler = Handler()
            try:
                cam.start_streaming(handler=handler,
                                    buffer_count=15, #jugar con este parametro
                                    allocation_mode=AllocationMode.AllocAndAnnounceFrame) #Jugar con este parametro
                input()
                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
                while True:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        cv2.destroyWindow(msg.format(cam.get_name()))
                        break

                    display = handler.get_image()
                    x, y, major, minor, phi, intensity = handler.get_info()
                    cv2.putText(display, f"x:{x:.0f}, y: {y:.0f}",(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
                    cv2.putText(display, f"major axis:{major:.2f} [um], minor axis: {minor:.2f} [um]",(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
                    cv2.putText(display, f"phi:{phi:.2f}, integrated intensity: {intensity}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
                    cv2.imshow(msg.format(cam.get_name()), display)
            finally:
                cam.stop_streaming()
    
if __name__ == '__main__':
    main()