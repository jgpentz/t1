import FileOptions from './FileOptions';

export default {
    component: FileOptions,
    title: 'FileOptions',
    decorators: [(story) => <div style={{ width: '300px' }}>{story()}</div>],
    tags: ['autodocs'],
};

export const Default = {
    args: {
        fname: 'File 1.s2p',
        sparams: ['S11', 'S12', 'S21', 'S22']
    },
};
