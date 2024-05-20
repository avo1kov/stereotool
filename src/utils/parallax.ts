// src/utils/parallax.ts
import cv from 'opencv.js';

export const calculateParallax = (img1: HTMLImageElement, img2: HTMLImageElement): number => {
  const mat1 = cv.imread(img1);
  const mat2 = cv.imread(img2);

  const stereo = new cv.StereoBM();
  const disparity = new cv.Mat();
  stereo.compute(mat1, mat2, disparity);

  const parallax = cv.mean(disparity)[0];
  return parallax;
};

export const findBestParallaxPair = (images: HTMLImageElement[]): [HTMLImageElement, HTMLImageElement] => {
  let bestPair: [HTMLImageElement, HTMLImageElement] = [images[0], images[1]];
  let bestParallax = -Infinity;

  for (let i = 0; i < images.length; i++) {
    for (let j = i + 1; j < images.length; j++) {
      const parallax = calculateParallax(images[i], images[j]);
      if (parallax > bestParallax) {
        bestParallax = parallax;
        bestPair = [images[i], images[j]];
      }
    }
  }

  return bestPair;
};
