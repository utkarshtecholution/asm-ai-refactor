import cv2
from threading import Thread

class CameraStreamConfigurator:
    def __init__(self, camera_id):
        """
        Initialize the camera configuration with a given camera ID.

        :param camera_id: ID of the camera to configure.
        """
        self.cam_id = camera_id
        self.cap = cv2.VideoCapture(self.cam_id)
        self.org_frame = None
        self.resized_frame = None
        self.stream_flag = False
        self.thread = None
        if not self.cap.isOpened():
            raise ValueError(f"Camera with ID {self.cam_id} could not be opened.")
    
    def decode_fourcc(self, v):
        """
        Decode a four-character code (FOURCC) integer to a string.

        :param v: FOURCC code in integer form.
        :return: Decoded FOURCC code as a string.
        """
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    def configure_resolution(self, width=640, height=480, fps=30, codec='MJPG'):
        """
        Configure the camera resolution, frames per second (FPS), and codec.

        :param width: Desired width of the video frame.
        :param height: Desired height of the video frame.
        :param fps: Desired frames per second.
        :param codec: Desired codec (default is 'MJPG').
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Set codec
        old_fourcc = self.decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))
        if self.cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            print(f"Codec changed from {old_fourcc} to {self.decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))}")
        else:
            print(f"Error: Could not change codec from {old_fourcc}.")

        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if self.cap.get(cv2.CAP_PROP_FPS) != fps:
            print(f"Warning: Could not set FPS to {fps}. Current FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        
        # Set frame width and height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width or self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != height:
            print(f"Warning: Could not set resolution to {width}x{height}. Current resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        # Print the current configuration
        self.print_camera_configuration()

    def print_camera_configuration(self):
        """
        Print the current camera configuration.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        codec = self.decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))
        
        print(f"Current camera configuration:\n"
              f"Codec: {codec}\n"
              f"FPS: {fps}\n"
              f"Resolution: {width}x{height}")

    def read_frames(self, resized_width: int, resized_height: int):
        while self.stream_flag:
            ret, frame = self.cap.read()
            if ret is False:
                continue
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            self.org_frame = frame
            self.resized_frame = cv2.resize(frame, (resized_width, resized_height), cv2.INTER_CUBIC)
        # if self.thread is not None:
        #     self.thread.join()
        self.thread = None

    def run_streaming_in_thread(self, resized_width, resized_height):
        self.thread = Thread(target=self.read_frames, args=(resized_width, resized_height))
        self.thread.start()
