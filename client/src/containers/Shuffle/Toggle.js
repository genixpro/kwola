import React from 'react';
import Button from '../../components/uielements/button';
import Icon from '../../components/uielements/icon/index.js';
const Toggle = ({
  clickHandler,
  text,
  icon,
  active,
  large,
  color,
  variant,
  className,
}) => {
  const iconClass = `${icon}`;

  return (
    <Button
      variant={variant}
      color={color}
      className={className}
      onClick={clickHandler}
    >
      {/* <MailIcon /> */}
      <Icon className="mateShuffleIcon" color="primary">
        {iconClass}
      </Icon>

      {text}
    </Button>
  );
};

export default Toggle;
