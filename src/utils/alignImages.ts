// src/utils/alignImages.ts
import cv from 'opencv.js';

export const alignImages = (images: HTMLImageElement[]): HTMLImageElement[] => {
  const alignedImages: HTMLImageElement[] = [];
  const refImg = images[0];
  const refMat = cv.imread(refImg);

  for (let i = 1; i < images.length; i++) {
    const img = images[i];
    const mat = cv.imread(img);

    const alignedMat = new cv.Mat();
    const warpMat = cv.getPerspectiveTransform(mat, refMat);
    cv.warpPerspective(mat, alignedMat, warpMat, new cv.Size(refMat.cols, refMat.rows));

    const alignedImg = new Image();
    cv.imshow(alignedImg, alignedMat);
    alignedImages.push(alignedImg);
  }

  return alignedImages;
};
