// src/App.tsx
import React, { useState } from 'react';
import Upload from './components/Upload';
import DepthControl from './components/DepthControl';
import { alignImages } from './utils/alignImages';
import { findBestParallaxPair } from './utils/parallax';
import { balanceColors } from './utils/colorBalance';
import { downloadImage } from './utils/download';

const App: React.FC = () => {
  const [images, setImages] = useState<HTMLImageElement[]>([]);
  const [bestPair, setBestPair] = useState<[HTMLImageElement, HTMLImageElement] | null>(null);
  const [depth, setDepth] = useState<number>(50);

  const handleFilesUploaded = (files: File[]) => {
    const imageElements = files.map((file) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      return img;
    });
    setImages(imageElements);
  };

  const processImages = () => {
    if (images.length < 2) {
      return;
    }

    const alignedImages = alignImages(images);
    const [img1, img2] = findBestParallaxPair(alignedImages);
    const [balancedImg1, balancedImg2] = balanceColors(img1, img2);
    setBestPair([balancedImg1, balancedImg2]);
  };

  const handleDownload = () => {
    if (bestPair) {
      const stereoImage = createStereoImage(bestPair[0], bestPair[1], depth);
      downloadImage(stereoImage, 'stereo.jpg');
    }
  };

  const createStereoImage = (img1: HTMLImageElement, img2: HTMLImageElement, depth: number): HTMLImageElement => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    const width = img1.width;
    const height = img1.height;
    canvas.width = width * 2;
    canvas.height = height;
    context.drawImage(img1, 0, 0, width, height);
    context.drawImage(img2, width + depth, 0, width, height);

    const stereoImage = new Image();
    stereoImage.src = canvas.toDataURL('image/jpeg');
    return stereoImage;
  };

  return (
    <div>
      <Upload onFilesUploaded={handleFilesUploaded} />
      <button onClick={processImages}>Process Images</button>
      {bestPair && <DepthControl depth={depth} onDepthChange={setDepth} />}
      {bestPair && <button onClick={handleDownload}>Download Stereo Image</button>}
    </div>
  );
};

export default App;
