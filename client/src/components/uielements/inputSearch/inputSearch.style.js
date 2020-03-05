import styled from 'styled-components';
import { palette } from 'styled-theme';
import WithDirection from '../../../settings/withDirection';
import Input from '../input';

const InputSearches = styled(Input)`
  > div {
    &:before,
    &:after {
      height: 1px;
    }

    &:before {
      background-color: ${palette('grey', 3)};
      ${'' /* background-color: transparent; */};
    }

    &:hover {
      &:before {
        height: 1px !important;
      }
    }

    input {
      &::-webkit-input-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 4)};
      }

      &:-moz-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 4)};
      }

      &::-moz-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 4)};
      }
      &:-ms-input-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 4)};
      }
    }
  }
`;

const InputSearch = WithDirection(InputSearches);

export { InputSearch };
