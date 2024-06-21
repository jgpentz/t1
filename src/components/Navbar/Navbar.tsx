import { ActionIcon, AppShell, Center, Code, Group, Tooltip, UnstyledButton } from "@mantine/core";
import { TbChartLine, TbLayoutSidebarLeftCollapse, TbLayoutSidebarLeftExpand, TbSettings } from "react-icons/tb";
import classes from './Navbar.module.css'
import FrfLogo from "../FrfLogo/FrfLogo";
import { useState } from "react";
import { NavbarLink, NavbarLinkData } from "../NavbarLink/NavbarLink";

// Navigation links
const links = [
    { link: '', label: 'S Parameters', icon: TbChartLine },
    // { link: '', label: 'Settings', icon: TbSettings },
];


// Type definition for the props passed into Navbar component
type NavbarProps = {
    collapsed: boolean;
    toggle: () => void;
};

export function Navbar({collapsed, toggle}: NavbarProps) {
    const [active, setActive] = useState<string>('S Parameters')

    return (
        <AppShell.Navbar p="sm" className="{classes.navbar}">
            {collapsed ? ( 
                /* Collapsed navbar only shows expand button, link icons, and version */
                <>
                    <Center>
                        <Tooltip label="Expand" position="right" transitionProps={{ duration: 0 }}>
                            <ActionIcon
                                onClick={toggle}
                                size="xl"
                                aria-label="Expand sidebar"
                                className={classes.collapse}
                                variant="transparent"
                            >
                                <TbLayoutSidebarLeftExpand className={classes.icon}/>
                            </ActionIcon>
                        </Tooltip>

                    </Center>

                    {/* Render each link that exists in the links variable*/}
                    <div className={classes.navbarMainC}>
                        {links.map((item, index) => {
                            const link_data: NavbarLinkData = {
                                link: item.link,
                                label: item.label,
                                icon: item.icon
                            }

                            return (
                                <NavbarLink 
                                    key={`${item.link}-${index}`}
                                    selected={active} 
                                    setSelected={setActive} 
                                    collapsed={collapsed} 
                                    link_data={link_data}
                                />
                            );
                        })}
                    </div>

                    {/* Semantic version */}
                    {/* TODO: This should be a variable that can be automatically updated */}
                    <div>
                        <Code>v0.0.1</Code>
                    </div>
                </>
            ) : (
                /* Full size navbar shows logo, links with icons and name, and version */
                <>
                    <Group gap={0}>
                        <FrfLogo width={170} height={50} />
                        <Tooltip label="Collapse" position="right" transitionProps={{ duration: 0 }}>
                            <ActionIcon
                                onClick={toggle}
                                size="xl"
                                aria-label="Collapse sidebar"
                                className={classes.collapse}
                                variant="transparent"
                            >
                                <TbLayoutSidebarLeftCollapse className={classes.icon}/>
                            </ActionIcon>
                        </Tooltip>
                    </Group>

                    {/* Render each link that exists in the links variable*/}
                    <div className={classes.navbarMain}>
                        {links.map((item, index) => {
                            const link_data: NavbarLinkData = {
                                link: item.link,
                                label: item.label,
                                icon: item.icon
                            }

                            return (
                                <NavbarLink 
                                    key={`${item.link}-${index}`}
                                    selected={active} 
                                    setSelected={setActive} 
                                    collapsed={collapsed} 
                                    link_data={link_data}
                                />
                            );
                        })}
                    </div>

                    {/* Semantic version */}
                    {/* TODO: This should be a variable that can be automatically updated */}
                    <div>
                        <Code>v0.0.1</Code>
                    </div>
                </>
            )}
        </AppShell.Navbar>
    );
}
