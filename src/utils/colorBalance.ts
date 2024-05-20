// src/utils/colorBalance.ts
import cv from 'opencv.js';

export const balanceColors = (img1: HTMLImageElement, img2: HTMLImageElement): [HTMLImageElement, HTMLImageElement] => {
  const mat1 = cv.imread(img1);
  const mat2 = cv.imread(img2);

  const mat1Lab = new cv.Mat();
  const mat2Lab = new cv.Mat();
  cv.cvtColor(mat1, mat1Lab, cv.COLOR_BGR2Lab);
  cv.cvtColor(mat2, mat2Lab, cv.COLOR_BGR2Lab);

  const lab1 = new cv.MatVector();
  const lab2 = new cv.MatVector();
  cv.split(mat1Lab, lab1);
  cv.split(mat2Lab, lab2);

  const l1 = lab1.get(0);
  const l2 = lab2.get(0);

  const l1Hist = new cv.Mat();
  const l2Hist = new cv.Mat();
  cv.calcHist([l1], [0], new cv.Mat(), l1Hist, [256], [0, 256]);
  cv.calcHist([l2], [0], new cv.Mat(), l2Hist, [256], [0, 256]);

  const l1HistCum = new cv.Mat();
  const l2HistCum = new cv.Mat();
  cv.normalize(l1Hist, l1HistCum, 0, 255, cv.NORM_MINMAX);
  cv.normalize(l2Hist, l2HistCum, 0, 255, cv.NORM_MINMAX);

  const lut = new cv.Mat(1, 256, cv.CV_8U);
  for (let i = 0; i < 256; i++) {
    const idx = l2HistCum.at(i);
    lut.data[i] = idx;
  }

  const balancedL1 = new cv.Mat();
  cv.LUT(l1, lut, balancedL1);

  const balancedImg1 = new Image();
  const balancedImg2 = new Image();
  cv.merge([balancedL1, lab1.get(1), lab1.get(2)], mat1Lab);
  cv.merge([l2, lab2.get(1), lab2.get(2)], mat2Lab);
  cv.cvtColor(mat1Lab, mat1, cv.COLOR_Lab2BGR);
  cv.cvtColor(mat2Lab, mat2, cv.COLOR_Lab2BGR);

  cv.imshow(balancedImg1, mat1);
  cv.imshow(balancedImg2, mat2);

  return [balancedImg1, balancedImg2];
};
