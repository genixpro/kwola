import React from 'react';
import { Route } from 'react-router-dom';
import { AnimatedRoute, spring } from 'react-router-transition';
import { siteConfig } from '../../settings';

function slide(val) {
  return spring(val, {
    stiffness: 300,
    damping: 30,
  });
}

const transitionProps = {
  atEnter: {
    opacity: 0,
    offset: -5,
  },
  atLeave: {
    opacity: slide(0),
    offset: 0,
  },
  atActive: {
    opacity: slide(1),
    offset: 0,
  },
  mapStyles: styles => ({
    opacity: styles.opacity,
    transform: `translateY(${styles.offset}%)`,
  }),
  className: 'router-transition',
};
export default props =>
  siteConfig.enableAnimatedRoute ? (
    <AnimatedRoute {...transitionProps} {...props} />
  ) : (
    <Route {...props} />
  );
