# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import time
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from typing import List
from nanosam.utils.owlvit import OwlVit
from nanosam.utils.predictor import Predictor
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="../../data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="../../data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--image_encoder_engine", type=str, default="../../data/owl_image_encoder_patch32.engine")
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    CAMERA_DEVICE = args.camera
    IMAGE_QUALITY = args.image_quality

    predictor = OwlPredictor(
            image_encoder_engine=args.image_encoder_engine
        )

    sam_predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )

    prompt_data = None

    def bbox2points(bbox):
        
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])

        point_labels = np.array([2, 3])
        return points, point_labels   

    def get_colors(count: int):
        cmap = plt.cm.get_cmap("rainbow", count)
        colors = []
        for i in range(count):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            colors.append(tuple(color))
        return colors


    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)


    async def handle_index_get(request: web.Request):
        logging.info("handle_index_get")
        return web.FileResponse("./index.html")


    async def websocket_handler(request):

        global prompt_data

        ws = web.WebSocketResponse()

        await ws.prepare(request)

        logging.info("Websocket connected.")

        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                logging.info(f"Received message from websocket.")
                if "prompt" in msg.data:
                    header, prompt = msg.data.split(":")
                    logging.info("Received prompt: " + prompt)
                    try:
                        prompt = prompt.strip("][()")
                        text = prompt.split(',')
                        text_encodings = predictor.encode_text(text)

                        prompt_data = {
                            "text": text,
                            "owl_encodings": text_encodings
                        }
                        logging.info("Set prompt: " + prompt)
                    except Exception as e:
                        print(e)
        finally:
            request.app['websockets'].discard(ws)

        return ws


    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY,
                        message='Server shutdown')


    async def detection_loop(app: web.Application):

        loop = asyncio.get_running_loop()

        logging.info("Opening camera.")

        camera = cv2.VideoCapture(CAMERA_DEVICE)

        logging.info("Loading predictor.")

        def _read_and_encode_image():

            re, image = camera.read()

            if not re:
                return re, None

            image_pil = cv2_to_pil(image)

            if prompt_data is not None:
                prompt_data_local = prompt_data
                t0 = time.perf_counter_ns()

                detections = predictor.predict(
                        image=image_pil, 
                        text=prompt_data_local['text'], 
                        text_encodings=prompt_data_local['owl_encodings'],
                        threshold=0.1,
                        pad_square=False
                    )

                sam_predictor.set_image(image_pil)

                if detections.boxes.size(0) != 0:
                    N = len(detections.labels)
                    for i in range(N):
                        bbox = detections.boxes[i]
                        box = [int(x) for x in bbox]
                        print(bbox)
                        print(box)
                        points, point_labels = bbox2points(box)
                        mask, _, _ = sam_predictor.predict(points, point_labels)
                        # draw mask
                        overlay_color = (0,255,0)
                        overlay = np.zeros_like(image,dtype=np.uint8)
                        overlay[mask[0, 0].detach().cpu() > 0] = overlay_color
                        alpha = 0.5
                        image = cv2.addWeighted(image,1-alpha,overlay,alpha,0)
                
             
                t1 = time.perf_counter_ns()
                dt = (t1 - t0) / 1e9

                # draw  box
                text = prompt_data_local['text']
                image = draw_owl_output(image, detections, text=text, draw_text=True)

                
                #image = draw_tree_output(image, detections, prompt_data_local['tree'])

            image_jpeg = bytes(
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
            )

            return re, image_jpeg

        while True:

            re, image = await loop.run_in_executor(None, _read_and_encode_image)
            
            if not re:
                break
            
            for ws in app["websockets"]:
                await ws.send_bytes(image)

        camera.release()


    async def run_detection_loop(app):
        try:
            task = asyncio.create_task(detection_loop(app))
            yield
            task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            await task


    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    web.run_app(app, host=args.host, port=args.port)