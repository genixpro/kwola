import React from 'react';
import styled from 'styled-components';
export default function CodeViewer({ children }) {
  return (
    <Container>
      <pre>{children}</pre>
    </Container>
  );
}

const Container = styled.div`
  font-family: monospace;
  padding: 30px;
  position: relative;
  overflow: hidden;
  background-color: #263238;
  color: #fff;
  pre {
    padding: 0 4px;
    padding-left: 15px;
    -moz-border-radius: 0;
    -webkit-border-radius: 0;
    border-radius: 0;
    border-width: 0;
    background: transparent;
    font-family: inherit;
    font-size: inherit;
    margin: 0;
    white-space: pre;
    word-wrap: normal;
    line-height: inherit;
    color: inherit;
    z-index: 2;
    position: relative;
    overflow: visible;
    -webkit-tap-highlight-color: transparent;
    -webkit-font-variant-ligatures: contextual;
    font-variant-ligatures: contextual;
  }
`;
