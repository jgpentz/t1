import { Dispatch, SetStateAction, useEffect, useRef, useState } from 'react';
import { Box, Container, Group, Text, rem } from '@mantine/core';
import { Dropzone, FileRejection } from '@mantine/dropzone';
import { TbBoxMargin, TbChartLine, TbFileUpload, TbGraph, TbX } from 'react-icons/tb';
import classes from './SparamGraph.module.css';
import { DataSet, SGraphDataLiteral, SGraphDataPoint, SparamFiles } from '@/pages/Sparams.page';
import Plot from 'react-plotly.js';
import { HighlightButton, HighlightModal, HighlightShape } from '../ModebarButtons/HighlightButton';
import { useDisclosure } from '@mantine/hooks';
import { FileDropzone } from '../FileDropzone/FileDropzone';

// Color-blind friendly color palette
const okabe_ito_colors: string[] = [
    "#000000", // Black
    "#E69F00", // Light orange
    "#56B4E9", // Light blue
    "#009E73", // Green
    "#F0E442", // Yellow
    "#0072B2", // Dark blue
    "#D55E00", // Dark orange
    "#CC79A7", // Pink

]

interface SparamGraphProps {
    fileData: Record<string, SparamFiles>;
    setFileData: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
    sparams: Record<string, SparamFiles>;
    setSparams: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
}

export function SparamGraph({fileData, setFileData, sparams, setSparams}: SparamGraphProps) {
    const [height, setHeight] = useState(window.innerHeight * 0.85);
    const [lineData, setLineData] = useState<any[]>([]);
	const [opened, {open, close}] = useDisclosure(false); // Disclosure used for highlight modal
    const highlightButton = HighlightButton(open); // highlight area of graph button
    const [highlights, setHighlights] = useState<HighlightShape[]>([]);
    const [plotKey, setPlotKey] = useState(0);
    const [dropzoneVisible, setDropzoneVisible] = useState(true); // Initially, dropzone is invisible
    const timer = useRef<number | NodeJS.Timeout | null>(null); // Timer to delay hiding dropzone

    /* Update the height to make our graph responsive to changes in window size */
    useEffect(() => {
        const handleResize = () => {
            setHeight(window.innerHeight * 0.85);
        };
        
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [window.innerHeight]);

    /* Assign all of the sparams to an array of lines to plot */
    useEffect(() => {
        const allSObjects: DataSet[] = [];

        // Get each sparam, assign a color, and store it in allSObjects
        let i = 0;
        for (const filename in sparams) {
            const fileData = sparams[filename];
            for (const key in fileData) {
                if (key.startsWith('s')) {
                    // Assign the line color and then append it to the lineData list
                    (fileData[key] as any).color = okabe_ito_colors[i % okabe_ito_colors.length]
                    allSObjects.push(fileData[key])
                    i += 1
                }
            }
        }

        setLineData(allSObjects);
    }, [sparams]);

    useEffect(() => {
        setPlotKey(prevKey => prevKey + 1);
    }, [highlights]);


    const handleLegendClick = (e: any) => {
        const [fname, sname] = e.data[e.curveNumber].name.split(" ");

        setSparams(prevSparams => {
            const updatedFileData = { ...prevSparams[fname] };
            const graphData = updatedFileData[sname] as unknown as SGraphDataLiteral;
            graphData.visible = !graphData.visible;
            return { ...prevSparams, [fname]: updatedFileData };
        });

        return false; // Prevent default legend click behavior
    };

    /* Handle when a file is dragged onto the window */
    const handleDragOver = () => {
        console.log('hey')
        if (!dropzoneVisible) {
            setDropzoneVisible(true); // Set dropzone visible when a file is dragged over
        }
        if (timer.current) {
            clearTimeout(timer.current as number); // Clear any existing timer
        }
    };

    /* Handle when a dragged file leaves the window */
    const handleDragLeave = () => {
        // Delay hiding dropzone by 200 milliseconds to prevent flickering
        timer.current = setTimeout(() => {
            // FIXME: When dragging a file onto the screen and then off the screen,
            // should the dropzone text only be turned off if sparams has data?
            if(fileData && Object.keys(fileData).length > 0){
                setDropzoneVisible(false);
            }
        }, 100);
    };

    return (
        <Container
            fluid
            w='100%'
            style={{ height: height }} 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
            <HighlightModal highlights={highlights} setHighlights={setHighlights} opened={opened} close={close} />
            <Plot
                key={plotKey}
                data={
                    [
                        ...lineData.map((s) => ({
                            x: s.data.map((item: any) => item.frequency),
                            y: s.data.map((item: any) => item.value),
                            type: 'scatter',
                            mode: 'lines',
                            name: s.name,
                            line: { color: s.color },
                            visible: s.visible ? true : 'legendonly',
                        }))
                    ]
                }
                layout={{
                    title: '<b>[click to edit title]</b>',
                    font: {
                        family: "Open Sans, Arial", // System must have at least one of these fonts installed
                        color: "#444",
                        size: 16,
                    },
                    autosize: true,
                    margin: { autoexpand: true},
                    hovermode: 'x',
                    xaxis: {
                        ticksuffix: 'G',
                        title: 'Hz',
                        type: 'linear',
                        autorange: true,
                        exponentformat: 'SI',
                        showgrid: true,
                        gridcolor: "#eee",
                        gridwidth: 1,
                        hoverformat: '.0f',
                        zeroline: true,
                        zerolinecolor: "#444",
                        zerolinewidth: 1,
                    },
                    yaxis: {
                        title: 'dB',
                        type: 'linear',
                        autorange: true,
                        exponentformat: 'SI',
                        showgrid: true,
                        gridcolor: "#eee",
                        gridwidth: 1,
                        hoverformat: '.1f',
                        zeroline: true,
                        zerolinecolor: "#444",
                        zerolinewidth: 1,
                    },
                    legend: { bgcolor: '#eee', xanchor: 'left', y: 1, yanchor: 'top' },
                    shapes: highlights,
                }}
                style={{width: '100%', height: '100%'}}
                useResizeHandler={true}
                config={{
                    displaylogo: false,
                    showTips: false,
                    editable: true,
                    modeBarButtonsToAdd: [highlightButton],
                }}
                onLegendClick={handleLegendClick}
            />

            { /* Dropzone is last in the container so react can render it on top*/ }
            {dropzoneVisible && 
                <FileDropzone 
                    fileData={fileData} 
                    setFileData={setFileData}
                    dropzoneVisible={dropzoneVisible}
                    setDropzoneVisible={setDropzoneVisible}
                />
            }
        </Container>
    );
}
