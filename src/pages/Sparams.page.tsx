import { Aside } from '@/components/Aside/Aside';
import { Navbar } from '@/components/Navbar/Navbar';
import { SparamGraph } from '@/components/SparamGraph/SparamGraph';
import { AppShell } from '@mantine/core';
import { useDisclosure, } from '@mantine/hooks';
import { Dispatch, SetStateAction, useState } from 'react';

export interface SGraphDataPoint {
    frequency: number;
    value: number;
}

export interface SGraphDataLiteral{
    name: string;
    visible: boolean;
    color: string
    data: SGraphDataPoint[];
}

export interface DataSet {
    m: number[];
    n: number[];
    frequency: number[];
    [key: string]: SGraphDataLiteral |  number[];
}

export interface SparamFiles {
    [filename: string]: DataSet;
}


interface SparamsPageProps {
    fileData: Record<string, SparamFiles>;
    setFileData: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
}

export function SparamsPage({ fileData, setFileData }: SparamsPageProps) {
    const [navCollapsed, { toggle: toggleNav }] = useDisclosure(false);
    const [sparams, setSparams] = useState<Record<string, SparamFiles>>({});

    return (
        <AppShell
            navbar={{
                width: (navCollapsed ? 80 : 240),
                breakpoint: 'xs',
                collapsed: { desktop: false, mobile: true}
            }}
            aside={{ 
                width: 300, 
                breakpoint: "md", 
                collapsed: { desktop: false, mobile: true }
            }}
        >
            <Navbar collapsed={navCollapsed} toggle={toggleNav}/>
            <AppShell.Main>
                <SparamGraph fileData={fileData} setFileData={setFileData} sparams={sparams} setSparams={setSparams}/>
            </AppShell.Main>
            <Aside sparams={sparams} setSparams={setSparams} />
        </AppShell>
    );
}
