# with face filter

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import os

class FaceBeardGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_models()

    def init_models(self):
        # Initialize base model
        self.base_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize inpainting model
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize face mesh detector for masking, and detecting the number of faces in the generated images
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,  # Set to 2 to detect if there's more than one face
            min_detection_confidence=0.5
        )

    def check_face_count(self, image):
        """
        Check if the image contains exactly one face.
        Returns True if exactly one face is detected, False otherwise.
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Check if faces were detected and count them
        if results.multi_face_landmarks is None:
            print("No faces detected in the image.")
            return False

        num_faces = len(results.multi_face_landmarks)
        if num_faces != 1:
            print(f"Found {num_faces} faces in the image. Skipping...")
            return False

        return True

    def create_face_mask(self, image, cutoff_y_rate=0.0):
        """
        Creates a mask for the lower portion of the face only.
        Modified to dilate downward only.
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            face_points = []

            min_y = min(lm.y for lm in landmarks)
            max_y = max(lm.y for lm in landmarks)
            cutoff_y = min_y + (max_y - min_y) * cutoff_y_rate

            for lm in landmarks:
                if lm.y > cutoff_y:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    face_points.append([x, y])

            if face_points:
                face_points = np.array(face_points, dtype=np.int32)
                hull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(mask, hull, 255)

                # Create a vertical-only kernel for downward dilation
                vertical_kernel = np.ones((200, 1), np.uint8)  # 200 pixels down, 1 pixel wide
                
                # Dilate the mask vertically
                mask = cv2.dilate(mask, vertical_kernel, iterations=1)

            return Image.fromarray(mask)

    def generate_versions(self, seed=42):
        # Generate base image with beard
        generator = torch.Generator(self.device).manual_seed(seed)

        base_prompt = (
            "professional studio headshot portrait of a young man with a slight beard, "
            "centered composition, facing directly forward, head and shoulders shot, "
            "neutral expression, sharp focus, high-end photography, perfect lighting, "
            "professional background, ultra high quality, detailed facial features, "
            "full head, head in the middle of the photo"
        )

        base_negative_prompt = (
            "bad quality, blurry, distorted, smile, teeth, side view, "
            "tilted head, asymmetrical face, multiple faces, watermark, "
            "text, out of frame, cropped, body parts missing, "
            "partial head, head not looking at the camera"
        )

        bearded = self.base_pipe(
            prompt=base_prompt,
            negative_prompt=base_negative_prompt,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        # Check if the generated image has exactly one face
        if not self.check_face_count(bearded):
            return None, None, None

        # Create mask for lower face
        mask = self.create_face_mask(bearded)

        # Generate clean-shaven version
        clean_prompt = (
            "professional studio headshot portrait of the same young man, "
            "completely clean-shaven face, smooth skin, centered composition, "
            "facing directly forward, head and shoulders shot, neutral expression, "
            "sharp focus, high-end photography, perfect lighting, "
            "professional background, ultra high quality"
        )

        clean_negative_prompt = (
            "beard, facial hair, stubble, mustache, smile, teeth, "
            "side view, tilted head, asymmetrical face, multiple faces, "
            "watermark, text, out of frame, cropped, body parts missing, "
            "different person, different face structure"
        )

        generator = torch.Generator(self.device).manual_seed(seed)

        clean_shaven = self.inpaint_pipe(
            prompt=clean_prompt,
            negative_prompt=clean_negative_prompt,
            image=bearded,
            mask_image=mask,
            num_inference_steps=75,
            generator=generator,
            strength=0.99,
            guidance_scale=9.0
        ).images[0]

        # Check if the clean-shaven version has exactly one face
        if not self.check_face_count(clean_shaven):
            return None, None, None

        return bearded, clean_shaven, mask

    def process_image(self, image_path, output_dir="output"):
        # Load image
        image = Image.open(image_path)

        # Check if image has exactly one face
        if not self.check_face_count(image):
            print(f"Skipping {image_path} due to invalid face count")
            return False

        # Create mask
        mask = self.create_face_mask(image)

        # Generate clean-shaven version
        clean_shaven = self.inpaint_pipe(
            prompt="professional headshot, completely clean-shaven face, smooth skin",
            negative_prompt="beard, facial hair, stubble, mustache, five o'clock shadow",
            image=image,
            mask_image=mask,
            num_inference_steps=100,
            strength=0.99,
            guidance_scale=9.0
        ).images[0]

        # Check if the generated image has exactly one face
        if not self.check_face_count(clean_shaven):
            print(f"Skipping {image_path} as generated image has invalid face count")
            return False

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, "original.png"))
        clean_shaven.save(os.path.join(output_dir, "clean_shaven.png"))
        mask.save(os.path.join(output_dir, "mask.png"))
        return True    