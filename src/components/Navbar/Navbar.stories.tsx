import { Navbar } from './Navbar';

export default {
    component: Navbar,
    title: 'Navbar',
    tags: ['autodocs'],
};

export const Default = {
    args: {
        collapsed: false,
    },
};

export const Collapsed = {
    args: {
        collapsed: true,
    },
};