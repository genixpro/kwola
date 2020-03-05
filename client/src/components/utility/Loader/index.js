import React from 'react';
import LoaderWrapper, { CircularProgress } from './style';

const Loader = () => (
  <LoaderWrapper>
    <CircularProgress size={50} />
  </LoaderWrapper>
);
export default Loader;
