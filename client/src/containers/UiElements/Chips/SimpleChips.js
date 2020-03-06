import React from 'react';
import Avatar from '../../../components/uielements/avatars';
import { Chip, Icon, Wrapper } from './chips.style';
import AvaterImg from '../../../images/user.jpg';

function handleRequestDelete() {
  alert('You clicked the delete icon.');
}

function handleClick() {
  alert('You clicked the Chip.');
}

function SimpleChips(props) {
  const { classes } = props;
  return (
    <Wrapper>
      <Chip label="Basic Chip" />
      <Chip
        avatar={<Avatar>MB</Avatar>}
        label="Clickable Chip"
        onClick={handleClick}
      />
      <Chip
        avatar={<Avatar src={AvaterImg} />}
        label="Deletable Chip"
        onDelete={handleRequestDelete}
        className={classes.chip}
      />
      <Chip
        avatar={
          <Avatar>
            <Icon>face</Icon>
          </Avatar>
        }
        label="Clickable Deletable Chip"
        onClick={handleClick}
        onDelete={handleRequestDelete}
      />
      <Chip
        label="Custom delete icon Chip"
        onClick={handleClick}
        onDelete={handleRequestDelete}
        deleteIcon={<Icon style={{ fontSize: 22 }}>done</Icon>}
      />
    </Wrapper>
  );
}
export default SimpleChips;
