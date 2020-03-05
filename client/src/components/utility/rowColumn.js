import React from 'react';
import Grids from '@material-ui/core/Grid';
import styled from 'styled-components';

const Grid = styled(Grids)`
  margin-top: 0;
  margin-bottom: 0;
`;

const Row = props => (
  <Grid container spacing={3} style={props.style}>
    {props.children}
  </Grid>
);
const Column = props => (
  <Grid
    item
    xs={props.xs}
    sm={props.sm}
    lg={props.lg}
    md={props.md}
    style={props.style}
  >
    {props.children}
  </Grid>
);
const HalfColumn = props => (
  <Column
    xs={props.xs ? props.xs : 12}
    lg={props.lg ? props.lg : 6}
    md={props.md ? props.md : 6}
    sm={props.sm ? props.sm : 12}
    style={props.style}
    {...props}
  />
);
const FullColumn = props => (
  <Column
    xs={props.xs ? props.xs : 12}
    lg={props.lg ? props.lg : 12}
    md={props.md ? props.md : 12}
    sm={props.sm ? props.sm : 12}
    style={props.style}
    {...props}
  />
);
const OneFourthColumn = props => (
  <Column
    xs={props.xs ? props.xs : 12}
    lg={props.lg ? props.lg : 3}
    md={props.md ? props.md : 3}
    sm={props.sm ? props.sm : 6}
    style={props.style}
    {...props}
  />
);

const OneThirdColumn = props => (
  <Column
    xs={props.xs ? props.xs : 12}
    lg={props.lg ? props.lg : 4}
    md={props.md ? props.md : 4}
    sm={props.sm ? props.sm : 6}
    style={props.style}
    {...props}
  />
);

const TwoThirdColumn = props => (
  <Column
    xs={props.xs ? props.xs : 12}
    lg={props.lg ? props.lg : 8}
    md={props.md ? props.md : 8}
    sm={props.sm ? props.sm : 6}
    style={props.style}
    {...props}
  />
);

export {
  Row,
  Column,
  HalfColumn,
  FullColumn,
  OneFourthColumn,
  TwoThirdColumn,
  OneThirdColumn,
};
