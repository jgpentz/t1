import { Dispatch, SetStateAction, useEffect, useState } from 'react';
import { AppShell, Flex, ScrollArea } from '@mantine/core';
import classes from './Aside.module.css'; // Import your CSS module for styling
import FileOptions from '../FileOptions/FileOptions';
import { SparamFiles } from '@/pages/Sparams.page';

interface FileDescriptor {
    fname: string;
    snames: string[];
}

interface AsideProps {
    sparams: Record<string, SparamFiles>;
    setSparams: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
}

export function Aside({sparams, setSparams}: AsideProps) {
    const [files, setFiles] = useState<FileDescriptor[]>([])

    // Build a list of file descriptors for each file in sparams, which consists of a filename
    // and the sparams
    useEffect(() => {
        const new_files: FileDescriptor[] = []; 

        // Loop through each file in the sparams object
        for (const filename in sparams) {
            // Create a file descriptor with the filename
            const fdesc: FileDescriptor = {
                fname: filename,
                snames: []
            };

            // Append any sparams to the file descriptor
            const fileData = sparams[filename];
            for (const key in fileData) {
                if (key.startsWith('s')) {
                    fdesc.snames.push(key)
                }
            }

            new_files.push(fdesc)
        }

        // Assign the new files
        setFiles(new_files);
    }, [sparams]);

    return (
        <AppShell.Aside>
            <AppShell.Section grow component={ScrollArea} scrollbars="y">
                <div className={classes.aside}>
                    {files.map((f, idx) => {
                        return (
                            <FileOptions 
                                key={`${f.fname}-${idx}`}
                                sparams={sparams} 
                                setSparams={setSparams} 
                                fname={f.fname} 
                                snames={f.snames} 
                            />
                        )
                    })}
                </div>
            </AppShell.Section>
        </AppShell.Aside>
    );
}
