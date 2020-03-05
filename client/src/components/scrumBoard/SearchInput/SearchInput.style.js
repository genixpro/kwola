import styled from 'styled-components';

export const SearchWrapper = styled.div`
  width: 300px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  position: relative;

  .MuiTextField-root {
    width: calc(100% - 32px);
  }

  @media only screen and (max-width: 767px) {
    width: 100%;
  }
`;

export const ClearButton = styled.button`
  margin-left: -1rem;
  cursor: pointer;
  border: 0;
  background-color: transparent;
  position: absolute;
  right: 0;
  top: auto;
  &:hover,
  &:focus {
    outline: none;
  }
`;
