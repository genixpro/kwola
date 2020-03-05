import styled from 'styled-components';
import Papersheet from '../../components/utility/papersheet';
import Tables from '../../components/uielements/table';

const Table = styled(Tables)`
  tr {
    td,
    th {
      white-space: nowrap;
    }
  }
`;

const Root = styled(Papersheet)`
  width: 100%;
  overflow-x: auto;
  box-sizing: border-box;
`;

export { Root, Table };
