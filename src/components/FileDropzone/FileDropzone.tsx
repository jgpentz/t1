import classes from './FileDropzone.module.css';
import { SparamFiles } from "@/pages/Sparams.page";
import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import { Box, Container, Group, Text, rem } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Dropzone, FileRejection } from '@mantine/dropzone';
import { TbBoxMargin, TbChartLine, TbFileUpload, TbGraph, TbX } from 'react-icons/tb';

interface DropzoneProps {
    fileData: Record<string, SparamFiles>;
    setFileData: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
    dropzoneVisible: boolean;
    setDropzoneVisible: Dispatch<SetStateAction<boolean>>;
}

export function FileDropzone({fileData, setFileData, dropzoneVisible, setDropzoneVisible}: DropzoneProps) {
    const [fileUploading, setFileUploading] = useState(false);

    /* Assign all of the sparams to an array of lines to plot */
    useEffect(() => {
        // Show dropzone when no data is present
        if (fileData === null || Object.keys(fileData).length === 0) {
            setDropzoneVisible(true);
        }

    }, [fileData]);

    /* Send the new files to the backend for processing */
    const processFileData = async (files: File[]) => {
        try {
            const formData = new FormData();

            // Append each file to the FormData object
            files.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch('http://localhost:8080/process_files', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                setFileUploading(false)
                if (fileData && Object.keys(fileData).length > 0){
                    setDropzoneVisible(false)
                }

                notifications.show({
                    color: 'red',
                    title: 'Bad network response',
                    message: '',
                    classNames: classes,
                })

                throw new Error('Network response was not ok');
            }

            const responseData: Record<string, SparamFiles> = await response.json();
            console.log('Successfully sent data to the backend', responseData);

            setDropzoneVisible(false)
            setFileUploading(false)

            // Append the new sparams data
            setFileData(prevSparams => ({
                ...prevSparams,
                ...responseData
            }));


        } catch (error) {
                console.error('Error sending data to the backend', error);
                setFileUploading(false)
                if (fileData && Object.keys(fileData).length > 0){
                    setDropzoneVisible(false)
                }

                notifications.show({
                    color: 'red',
                    title: 'Failed to upload files',
                    message: '',
                    classNames: classes,
                })
        }
    };
    console.log('---filed start')
    console.log(fileData)
    console.log('---filed end')

    /* Send file data to backend, then store the processed data in sparams list */
    const getFileData = (files: File[]) => {
        const newFiles: File[] = []
        const acceptedFiles = new Set()
        const regex = /^s\d+p$/;

        // Filter files that match the regex pattern
        const filteredFiles = files.filter(file => {
            const fname = file.name.split('.');
            const fext = fname[fname.length - 1].toLowerCase();
            return regex.test(fext);
        });

        filteredFiles.forEach(file => {
            acceptedFiles.add(file.name)
            newFiles.push(file)
        });

        // Notify of bad file extensions
        const rejectedFiles: string[] = []
        files.forEach(file => {
            if (!acceptedFiles.has(file.name)) {
                rejectedFiles.push(file.name)
            }
        })
        if (rejectedFiles.length) {
            notifications.show({
                color: 'red',
                title: 'Bad file extension',
                message: rejectedFiles.join(', '),
                classNames: classes,
            })
        }

        // Only send files to backend if there were files with a snp extension
        if (newFiles.length > 0) {
            processFileData(newFiles);
        } else {
            setFileUploading(false)
            // Have to set the dropzone back to visible if there are no sparams
            if (fileData === null || Object.keys(fileData).length === 0) {
                setDropzoneVisible(true);
            } else {
                setDropzoneVisible(false);
            }
        }
    }

    /* Handle file drop */
    const handleDrop = (files: File[]) => {
        console.log('received files', files);
        setFileUploading(true)
        getFileData(files)
    };

    /* Handle rejected file types */
    const handleReject = (files: FileRejection[]) => {
        console.log('rejected files', files);

        setFileUploading(false)
        if (fileData && Object.keys(fileData).length > 0) {
            setDropzoneVisible(false)
        }

        const fnames: string[] = [] 
        files.forEach(file => {
            fnames.push(file.file.name)
        })

        notifications.show({
            color: 'red',
            title: 'Bad file extension',
            message: fnames.join(', '),
            classNames: classes,
        })
    };

    console.log(dropzoneVisible)

    return (
        <Dropzone
            onDrop={handleDrop}
            onReject={handleReject}
            loading={fileUploading}
            maxSize={5 * 1024 ** 2}
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                backgroundColor: 'rgba(255, 255, 255, 0.7)',
                display: 'flex',
                justifyContent: 'center',
                paddingTop: '15%',
            }}
        >
            <Group style={{ pointerEvents: 'none' }}>
                <Dropzone.Accept>
                    <TbFileUpload
                        style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-blue-6)' }}
                    />
                </Dropzone.Accept>
                <Dropzone.Reject>
                    <TbX style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-red-6)' }} />
                </Dropzone.Reject>
                <Dropzone.Idle>
                    <TbGraph
                        style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-dimmed)' }}
                    />
                </Dropzone.Idle>
                <div>
                    <Text size="xl" inline>
                        Drag SnP files here or click to select files
                    </Text>
                    <Text size="sm" inline mt={7}>
                        Attach as many files as you like
                    </Text>
                </div>
            </Group>
        </Dropzone>
    );
}
