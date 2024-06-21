import '@mantine/core/styles.css';
import { MantineProvider } from '@mantine/core';
import { Router } from './Router';
import { theme } from './theme';
import { Notifications } from '@mantine/notifications';
import { useState } from 'react';

export default function App() {
    const [fileData, setFileData] = useState(null);

    return (
        <MantineProvider theme={theme}>
            <Notifications position="top-right" zIndex={1000}/>
            <Router fileData={fileData as any} setFileData={setFileData as any}/>
        </MantineProvider>
    );
}
