// src/utils/download.ts
import { saveAs } from 'file-saver';

export const downloadImage = (image: HTMLImageElement, filename: string) => {
  image.onload = () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        saveAs(blob, filename);
      }
    }, 'image/jpeg');
  };
};
