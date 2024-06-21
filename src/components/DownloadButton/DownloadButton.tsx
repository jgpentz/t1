import React, { useRef } from 'react';
import { toPng } from 'html-to-image';
import { ActionIcon, Tooltip } from '@mantine/core';
import { TbCamera } from 'react-icons/tb';
import classes from './DownloadButton.module.css';

interface DownloadButtonProps {
  chartRef: React.RefObject<HTMLDivElement>;
}

export function DownloadButton({chartRef}: DownloadButtonProps) {
  const downloadChart = async () => {
    if (chartRef.current === null) {
      return;
    }

    const dataUrl = await toPng(chartRef.current, { cacheBust: true });
    const link = document.createElement('a');
    link.download = 'chart.png';
    link.href = dataUrl;
    link.click();
  };
  return (
    <Tooltip label="Download plot as png">
      <ActionIcon className={classes.icon} size="xl" variant="transparent" onClick={downloadChart}>
          <TbCamera style={{ width: 32, height: 32 }}/>
      </ActionIcon>
    </Tooltip>
  )
};
