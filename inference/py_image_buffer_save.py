# -----------------------------------------------------------------------------
# Copyright (c) 2021, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import os  # os.getcwd()
import time
from pathlib import Path

import numpy as np  # pip install numpy
from PIL import Image as PIL_Image  # pip install Pillow
import cv2
from arena_api.system import system
# from loguru import logger

def create_devices_with_tries():
    """
    This function waits for the user to connect a device before raising
    an exception
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            # logger.debug(
            #     f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
            #     f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                # logger.debug(f'{sec_count + 1 } seconds passed ',
                #       '.' * sec_count, end='\r')
            tries += 1
        else:
            # logger.debug(f'Created {len(devices)} device(s)')
            return devices
    else:
        # logger.error(f'No device found! Please connect a device and run '
                        # f'the example again.')
        pass

def create_device():
    # Create a device
    devices = create_devices_with_tries()
    device = devices[0]
    # logger.debug(f'Device used in the example:\n\t{device}')

    # Get device stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    # Enable stream auto negotiate packet size
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

    # Enable stream packet resend
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    # Get/Set nodes -----------------------------------------------------------
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])

    # Nodes
    # logger.debug('Setting Width to its maximum value')
    nodes['Width'].value = nodes['Width'].max

    # logger.debug('Setting Height to its maximum value')
    height = nodes['Height']
    height.value = height.max

    # Set pixel format to Mono8, most cameras should support this pixel format
    pixel_format_name = 'Mono8'
    # logger.debug(f'Setting Pixel Format to {pixel_format_name}')
    nodes['PixelFormat'].value = pixel_format_name

    # Grab and save an image buffer -------------------------------------------
     
    return device

def start_stream(device):
    # logger.debug('Starting stream')
    device.start_stream(1)
    
    return device
    
def grab_img(device, path):        
    # Optional args
    # logger.debug('Grabbing an image buffer')

    image_buffer = device.get_buffer()
    
    image_only_data = None
    if image_buffer.has_chunkdata:
        # logger.debug("has chunk")
    # 8 is the number of bits in a byte
        bytes_pre_pixel = int(image_buffer.bits_per_pixel / 8)

        image_size_in_bytes = image_buffer.height * \
            image_buffer.width * bytes_pre_pixel

        image_only_data = image_buffer.data[:image_size_in_bytes]
    else:
        image_only_data = image_buffer.data

    nparray = np.asarray(image_only_data, dtype=np.uint8)
    # Reshape array for pillow
    nparray_reshaped = nparray.reshape((
        image_buffer.height,
        image_buffer.width
    ))
    
    # Save image
    # png_name = path + '.png'
    png_array = PIL_Image.fromarray(nparray_reshaped)
    # flip
    # png_array_
    png_array = png_array.transpose(PIL_Image.ROTATE_180)
    # png_array.save(png_name)
    # logger.info(f'Saved image path is: {Path(os.getcwd()) / png_name}')
    device.requeue_buffer(image_buffer)
    image_array = cv2.rotate(nparray_reshaped, cv2.ROTATE_180)
    return image_array # png_array,

def drop_buffer(device):        
    # Optional args
    # logger.debug('Grabbing an image buffer')

    image_buffer = device.get_buffer()
    
    device.requeue_buffer(image_buffer)


def destroy_device():
    system.destroy_device()
    # logger.debug('Destroyed all created devices')
