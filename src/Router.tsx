import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { SparamsPage } from './pages/Sparams.page';
import { Dispatch, SetStateAction } from 'react';

interface RouterProps {
    fileData: Record<string, string>;
    setFileData: Dispatch<SetStateAction<Record<string, string>>>;
}

export function Router({ fileData, setFileData }: RouterProps) {
    const router = createBrowserRouter([
        {
            path: '/',
            element: <SparamsPage fileData={fileData as any} setFileData={setFileData as any} />,
        },
    ]);

    return <RouterProvider router={router} />;
}
