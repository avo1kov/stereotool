// src/components/Upload.tsx
import React from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadProps {
  onFilesUploaded: (files: File[]) => void;
}

const Upload: React.FC<UploadProps> = ({ onFilesUploaded }) => {
  const { getRootProps, getInputProps } = useDropzone({
    accept: 'image/*',
    onDrop: (acceptedFiles) => onFilesUploaded(acceptedFiles),
  });

  return (
    <div {...getRootProps({ className: 'dropzone' })} style={{ border: '2px dashed #cccccc', padding: '20px', textAlign: 'center' }}>
      <input {...getInputProps()} />
      <p>Drag 'n' drop some files here, or click to select files</p>
    </div>
  );
};

export default Upload;
