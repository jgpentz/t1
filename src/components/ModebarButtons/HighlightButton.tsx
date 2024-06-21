import React, { Dispatch, SetStateAction, useRef, useState } from 'react';
import { Accordion, ActionIcon, Button, Input, Modal, Text, Container } from '@mantine/core';
import Plotly from 'plotly.js-basic-dist';
import { MdAdd, MdCheck, MdClose } from 'react-icons/md';
import { M } from 'vite/dist/node/types.d-aGj9QkWt';

export interface HighlightShape {
	type?: 'line' | 'circle' | 'rect' | 'path';
	xref?: string;
	yref?: string;
	x0?: string | number;
	y0?: string | number;
	x1?: string | number;
	y1?: string | number;
	path?: string;
	line?: {
		color?: string;
		width?: number;
		dash?: 'solid' | 'dot' | 'dash' | 'longdash' | 'dashdot' | 'longdashdot';
	}
	fillcolor?: string;
	opacity?: string | number;
	layer?: 'below' | 'above' | 'between';
}

const NewHighlightShape = (): HighlightShape => (
	{
		type: 'rect', 
		xref:'x',
		yref:'y',
		x0: 0,
		y0: 0,
		x1: 0,
		y1: 0,
		fillcolor:'#d3d3d3',
		opacity: 0.4,
		line: {
			width: 0
		},
		layer: "between",
	}
)

interface HighlightButtonProps {
	opened: boolean,
	close: () => void,
	highlights: HighlightShape[],
	setHighlights: Dispatch<SetStateAction<HighlightShape[]>>
}

export function HighlightModal({opened, close, highlights, setHighlights}: HighlightButtonProps) {
	const [editingNew, setEditingNew] = useState<Boolean>(false)
	const [newHighlight, setNewHighlight] = useState<HighlightShape>(NewHighlightShape())

	const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>, idx: number, field: keyof HighlightShape) => {
		const value = event.target.value;
		setHighlights(prevHighlights => {
			const updatedHighlights = [...prevHighlights];
			updatedHighlights[idx] = {
				...updatedHighlights[idx],
				[field]: value
			};
			return updatedHighlights;
		});
	};

	const HandleNewHighlightChange = (event: React.ChangeEvent<HTMLInputElement>, field: keyof HighlightShape) => {
		const value = event.target.value;
		setNewHighlight(prevNewHighlight => ({
			...prevNewHighlight,
			[field]: value
		}));
	};

	const addHighlight = () => {
		setHighlights(prevHighlights => {
			const updatedHighlights = [...prevHighlights];
			updatedHighlights.push(newHighlight)
			return updatedHighlights;
		})
		setNewHighlight(NewHighlightShape())
		setEditingNew(false)
	}

	const discardNewHighlight = () => {
		setNewHighlight(NewHighlightShape())
		setEditingNew(false)
	}

	const deleteHighlight = (idx: number) => {
		setHighlights(prevHighlights => prevHighlights.filter((_, index) => index !== idx));
	};

	function formatNumber(value: number | string | undefined | null): string {
		if (value === undefined || value === null || value === '') {
			return '';
		}

		// If value is a string, attempt to convert it to a number
		const parsedValue = typeof value === 'string' ? parseFloat(value) : value;

		// Check if parsedValue is NaN (not a number)
		if (isNaN(parsedValue)) {
			return '';
		}

		return parsedValue.toFixed(2);
	}
	const items = highlights.map((highlight, idx) => (
		<div key={`${highlight.type}-${idx}`} style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
			<Accordion.Item value={`${highlight.type}-${idx}`} style={{ flexGrow: 1 }}>
				<Accordion.Control>
					<div style={{ display: 'flex', gap: '1rem'}}>
						<Text>{`x: [${formatNumber(highlight.x0)}, ${formatNumber(highlight.x1)}]`}</Text>
						<Text>{`y: [${formatNumber(highlight.y0)}, ${formatNumber(highlight.y1)}]`}</Text>
					</div>
				</Accordion.Control>
				<Accordion.Panel>
					<div style={{ display: 'flex', gap: '1rem' }}>
						<Input.Wrapper label='x0'>
							<Input value={String(highlight.x0)} onChange={(e) => handleInputChange(e, idx, 'x0')} />
						</Input.Wrapper>
						<Input.Wrapper label='x1'>
							<Input value={String(highlight.x1)} onChange={(e) => handleInputChange(e, idx, 'x1')} />
						</Input.Wrapper>
					</div>
					<div style={{ marginTop: '1rem', display: 'flex', gap: '1rem' }}>
						<Input.Wrapper label='y0'>
							<Input value={String(highlight.y0)} onChange={(e) => handleInputChange(e, idx, 'y0')} />
						</Input.Wrapper>
						<Input.Wrapper label='y1'>
							<Input value={String(highlight.y1)} onChange={(e) => handleInputChange(e, idx, 'y1')} />
						</Input.Wrapper>
					</div>
				</Accordion.Panel>
			</Accordion.Item>
			<ActionIcon onClick={() => deleteHighlight(idx)} style={{ marginLeft: '1rem', marginTop: '0.75rem'}}>
				<MdClose />
			</ActionIcon>
		</div>
	));

	return (
		<>
			<Modal opened={opened} onClose={close} title="Highlights">
				{editingNew ? 
					(
						<Container mb="1rem">
							<div style={{ display: 'flex', gap: '1rem'}}>
								<Input.Wrapper label='x0'>
									<Input value={String(newHighlight.x0)} onChange={(e) => HandleNewHighlightChange(e, 'x0')}/>
								</Input.Wrapper>
								<Input.Wrapper label='x1'>
									<Input value={String(newHighlight.x1)} onChange={(e) => HandleNewHighlightChange(e, 'x1')} />
								</Input.Wrapper>
							</div>
							<div style={{ marginTop: '1rem', display: 'flex', gap: '1rem'}}>
								<Input.Wrapper label='y0'>
									<Input value={String(newHighlight.y0)} onChange={(e) => HandleNewHighlightChange(e, 'y0')} />
								</Input.Wrapper>
								<Input.Wrapper label='y1'>
									<Input value={String(newHighlight.y1)} onChange={(e) => HandleNewHighlightChange(e, 'y1')} />
								</Input.Wrapper>
							</div>

							<div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem'}}>
								<Button color='green' leftSection={<MdCheck size={20}/>} onClick={addHighlight}>
									Add
								</Button>
								<Button color='red' leftSection={<MdClose size={20}/>} onClick={discardNewHighlight}>
									Discard
								</Button>
							</div>
						</Container>
					) : (
						<div style={{ marginTop: '0.5rem', marginBottom: '1rem', display: 'flex', gap: '0.5rem'}}>
							<ActionIcon onClick={() => setEditingNew(true)}>
								<MdAdd size={24}/>
							</ActionIcon>
							<Text>Add New</Text>
						</div>
					)
				}
				<Accordion variant='contained'>
					{items}
				</Accordion>
			</Modal>
		</>
	)
}

export function HighlightButton(open: () => void) {

	const highlightButton = {
		name: 'Highlight Area',
		icon: Plotly.Icons.selectbox,
		click: () => {
			open();
		},
	};

	return highlightButton;
};
