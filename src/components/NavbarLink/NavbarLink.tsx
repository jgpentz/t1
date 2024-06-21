import {  Center, Tooltip, UnstyledButton } from "@mantine/core";
import { IconType } from "react-icons";
import classes from './NavbarLink.module.css'

// Extend NavbarLinkData to include active and onClick properties
interface CollapsedData extends NavbarLinkData{
    active?: boolean;
    onClick?(event: React.MouseEvent<HTMLButtonElement>): void;
};

/* When the navbar is collapsed, the links become icons, with a tooltip
that displays the link name
*/
function CollapsedLink({ link, icon: Icon, label, active, onClick }: CollapsedData) {
  return (
    <Center>
        <Tooltip label={label} position="right" transitionProps={{ duration: 0 }}>
            <UnstyledButton onClick={onClick} className={classes.linkCollapsed} data-active={active || undefined}>
                <Icon className={classes.icon} data-active={active || undefined} />
            </UnstyledButton>
        </Tooltip>
    </Center>
  );
}


// Interface to define the structure of NavbarLinkData
export interface NavbarLinkData {
    link: string; // URL link for the navigation item
    label: string; // Display label for the navigation item
    icon: IconType; // Icon component for the navigation item
}

// Type definition for the props of NavbarLink component
type NavbarLinkProps = {
    selected: string; // Currently selected navigation item label
    setSelected: any; // Function to update the selected navigation item
    collapsed: boolean; // Boolean to determine if the navbar is collapsed
    link_data: NavbarLinkData; // Data for the navigation item
};

export function NavbarLink({selected, setSelected, collapsed, link_data}: NavbarLinkProps) {
    return (
        <>
            {collapsed ? (
                // Render the CollapsedLink component if the navbar is collapsed
                <CollapsedLink
                    link={link_data.link}
                    icon={link_data.icon}
                    label={link_data.label}
                    active={link_data.label === selected}
                    onClick={(event) => {
                        event.preventDefault();
                        setSelected(link_data.label)
                    }}
                />
            ) : (
                // Render an anchor tag if the navbar is not collapsed
                <a
                    className={classes.link}
                    data-active={link_data.label === selected || undefined}
                    href={link_data.link}
                    key={link_data.label}
                    onClick={(event) => {
                        event.preventDefault();
                        setSelected(link_data.label)
                    }}
                >
                    {/* Render the icon and label inside the anchor tag */}
                    <link_data.icon className={classes.linkIcon} data-expanded />
                    <span>{link_data.label}</span>
                </a>
            )}
        </>
    )
}