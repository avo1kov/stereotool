// src/components/DepthControl.tsx
import React from 'react';

interface DepthControlProps {
  depth: number;
  onDepthChange: (depth: number) => void;
}

const DepthControl: React.FC<DepthControlProps> = ({ depth, onDepthChange }) => {
  return (
    <div>
      <label>Depth: {depth}</label>
      <input
        type="range"
        min="0"
        max="100"
        value={depth}
        onChange={(e) => onDepthChange(Number(e.target.value))}
      />
    </div>
  );
};

export default DepthControl;
